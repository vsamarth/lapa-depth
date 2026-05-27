import bisect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class Stage252DatasetModel4(Dataset):
    """
    Dataset for Model 4.

    Pipeline:
        depth1 + z_rgb_features -> z_depth_feature

    This dataset does NOT return z_depth_indices.

    Expected z_depth JSONL per-line format:
        {
            "id": "videoid_imgxxxx",
            "image": "/path/to/depth.png",
            ...
        }

    Expected RGB feature manifest:
        {
            "total_samples": ...,
            "parts": [
                {
                    "path": "/path/to/z_rgb_feature_part.pt",
                    "num_samples": 8192
                }
            ]
        }

    Expected Stage-1 depth feature manifest:
        {
            "total_samples": ...,
            "parts": [
                {
                    "path": "/path/to/z_depth_feature_part.pt",
                    "num_samples": 8192
                }
            ]
        }

    Expected RGB .pt part format:
        dict with one of:
            "z_rgb_features", "rgb_features", "features", "z_features"

    Expected depth feature .pt part format:
        dict with one of:
            "z_depth_feature", "z_depth_features",
            "depth_feature", "depth_features",
            "features", "z_features"

    Returns:
        {
            "depth1": FloatTensor [C, H, W],
            "z_rgb_features": FloatTensor [4096],
            "z_depth_feature": FloatTensor [D] or [L, D],
            "id": str,
            "depth1_path": str
        }
    """

    def __init__(
        self,
        z_depth_path: Union[str, Path],
        z_rgb_feature_manifest: Union[str, Path],
        z_depth_feature_manifest: Union[str, Path],
        image_size: Union[int, Sequence[int]] = 256,
        *,
        repeat_depth_to_3ch: bool = True,
        depth_scale: float = 65535.0,
        strict: bool = False,
        check_length_alignment: bool = True,
        rgb_feature_key: Optional[str] = None,
        depth_feature_key: Optional[str] = None,
        keep_z_rgb_indices: bool = False,
        check_id_alignment: bool = False,
    ):
        super().__init__()

        self.z_depth_path = Path(z_depth_path)
        self.z_rgb_feature_manifest = Path(z_rgb_feature_manifest)
        self.z_depth_feature_manifest = Path(z_depth_feature_manifest)

        if not self.z_depth_path.exists():
            raise FileNotFoundError(f"z_depth_path not found: {self.z_depth_path}")
        if not self.z_rgb_feature_manifest.exists():
            raise FileNotFoundError(f"z_rgb_feature_manifest not found: {self.z_rgb_feature_manifest}")
        if not self.z_depth_feature_manifest.exists():
            raise FileNotFoundError(f"z_depth_feature_manifest not found: {self.z_depth_feature_manifest}")

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = tuple(image_size)

        self.repeat_depth_to_3ch = bool(repeat_depth_to_3ch)
        self.depth_scale = float(depth_scale)
        self.strict = bool(strict)
        self.check_length_alignment = bool(check_length_alignment)
        self.rgb_feature_key = rgb_feature_key
        self.depth_feature_key = depth_feature_key
        self.keep_z_rgb_indices = bool(keep_z_rgb_indices)
        self.check_id_alignment = bool(check_id_alignment)

        self.resize_depth = T.Resize(
            self.image_size,
            interpolation=T.InterpolationMode.NEAREST,
        )

        self.items: List[Dict[str, Any]] = []

        self.rgb_feature_parts: List[Dict[str, Any]] = []
        self.rgb_feature_cum_counts: List[int] = []

        self.depth_feature_parts: List[Dict[str, Any]] = []
        self.depth_feature_cum_counts: List[int] = []

        self._cached_rgb_part_idx: Optional[int] = None
        self._cached_rgb_part_data: Optional[Dict[str, Any]] = None

        self._cached_depth_part_idx: Optional[int] = None
        self._cached_depth_part_data: Optional[Dict[str, Any]] = None

        self._load_depth_jsonl_metadata()
        self._load_manifest(
            manifest_path=self.z_rgb_feature_manifest,
            target_parts=self.rgb_feature_parts,
            target_cum_counts=self.rgb_feature_cum_counts,
            name="RGB feature",
        )
        self._load_manifest(
            manifest_path=self.z_depth_feature_manifest,
            target_parts=self.depth_feature_parts,
            target_cum_counts=self.depth_feature_cum_counts,
            name="Depth feature",
        )

        if self.check_length_alignment:
            self._check_length_alignment()

        if len(self.items) == 0:
            raise RuntimeError(f"No valid samples loaded from z_depth_path={self.z_depth_path}")

    def __len__(self) -> int:
        return len(self.items)

    def _load_depth_jsonl_metadata(self) -> None:
        """
        Load only id and depth image path.
        Ignore delta / z_depth_indices completely.
        """
        with self.z_depth_path.open("r", encoding="utf-8") as fdep:
            for line_no, line_depth in enumerate(fdep, start=1):
                line_depth = line_depth.strip()
                if not line_depth:
                    continue

                try:
                    depth_item = json.loads(line_depth)
                    sample = self._build_depth_metadata_item(depth_item, line_no)
                    self.items.append(sample)
                except Exception as e:
                    if self.strict:
                        raise
                    print(f"[Stage252DatasetModel4] skip depth line {line_no}: {e}")

    def _build_depth_metadata_item(self, depth_item: Dict[str, Any], line_no: int) -> Dict[str, Any]:
        depth_id = str(depth_item.get("id", ""))

        depth_path = self._get_first_existing_key(
            depth_item,
            ["depth1_path", "depth1", "depth_path", "image", "depth_image"],
        )

        if not Path(depth_path).exists():
            raise FileNotFoundError(f"line {line_no}: depth path does not exist: {depth_path}")

        return {
            "id": depth_id,
            "depth1_path": str(depth_path),
            "depth_image": depth_item.get("image", None),
        }

    def _load_manifest(
        self,
        manifest_path: Path,
        target_parts: List[Dict[str, Any]],
        target_cum_counts: List[int],
        name: str,
    ) -> None:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        parts = manifest.get("parts", [])
        if not isinstance(parts, list) or len(parts) == 0:
            raise RuntimeError(f"No parts found in {name} manifest: {manifest_path}")

        total = 0
        for part in parts:
            part = dict(part)

            if "path" not in part:
                raise KeyError(f"{name} manifest part missing 'path': {part.keys()}")
            if "num_samples" not in part:
                raise KeyError(f"{name} manifest part missing 'num_samples': {part.keys()}")

            part_path = Path(part["path"])
            if not part_path.exists():
                raise FileNotFoundError(f"{name} .pt file not found: {part_path}")

            total += int(part["num_samples"])
            target_parts.append(part)
            target_cum_counts.append(total)

        declared_total = manifest.get("total_samples", None)
        if declared_total is not None and int(declared_total) != total:
            msg = f"{name} manifest total_samples={declared_total}, but sum(parts.num_samples)={total}"
            if self.strict:
                raise RuntimeError(msg)
            print(f"[Stage252DatasetModel4] Warning: {msg}")

    def _check_length_alignment(self) -> None:
        n_depth_jsonl = len(self.items)
        n_rgb_features = self.rgb_feature_cum_counts[-1] if self.rgb_feature_cum_counts else 0
        n_depth_features = self.depth_feature_cum_counts[-1] if self.depth_feature_cum_counts else 0

        if not (n_depth_jsonl == n_rgb_features == n_depth_features):
            msg = (
                "Length mismatch: "
                f"z_depth_jsonl={n_depth_jsonl}, "
                f"z_rgb_features={n_rgb_features}, "
                f"z_depth_features={n_depth_features}. "
                "Training assumes all sources are in the same sample order."
            )
            if self.strict:
                raise RuntimeError(msg)
            print(f"[Stage252DatasetModel4] Warning: {msg}")

    def _global_to_local(self, index: int, cum_counts: List[int]):
        part_idx = bisect.bisect_right(cum_counts, index)
        prev_end = 0 if part_idx == 0 else cum_counts[part_idx - 1]
        local_idx = index - prev_end
        return part_idx, local_idx

    def _load_rgb_part(self, part_idx: int) -> Dict[str, Any]:
        if self._cached_rgb_part_idx == part_idx and self._cached_rgb_part_data is not None:
            return self._cached_rgb_part_data

        part_path = self.rgb_feature_parts[part_idx]["path"]
        data = torch.load(part_path, map_location="cpu")
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict in RGB feature part {part_path}, got {type(data)}")

        self._cached_rgb_part_idx = part_idx
        self._cached_rgb_part_data = data
        return data

    def _load_depth_feature_part(self, part_idx: int) -> Dict[str, Any]:
        if self._cached_depth_part_idx == part_idx and self._cached_depth_part_data is not None:
            return self._cached_depth_part_data

        part_path = self.depth_feature_parts[part_idx]["path"]
        data = torch.load(part_path, map_location="cpu")
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict in depth feature part {part_path}, got {type(data)}")

        self._cached_depth_part_idx = part_idx
        self._cached_depth_part_data = data
        return data

    def _find_rgb_feature_key(self, data: Dict[str, Any]) -> str:
        if self.rgb_feature_key is not None:
            if self.rgb_feature_key not in data:
                raise KeyError(
                    f"Requested rgb_feature_key='{self.rgb_feature_key}' not found. "
                    f"Available keys: {list(data.keys())}"
                )
            return self.rgb_feature_key

        candidates = ["z_rgb_features", "rgb_features", "features", "z_features"]
        for key in candidates:
            if key in data:
                return key

        raise KeyError(f"Cannot find RGB feature key. Tried {candidates}. Available keys: {list(data.keys())}")

    def _find_depth_feature_key(self, data: Dict[str, Any]) -> str:
        if self.depth_feature_key is not None:
            if self.depth_feature_key not in data:
                raise KeyError(
                    f"Requested depth_feature_key='{self.depth_feature_key}' not found. "
                    f"Available keys: {list(data.keys())}"
                )
            return self.depth_feature_key

        candidates = [
            "z_depth_feature",
            "z_depth_features",
            "depth_feature",
            "depth_features",
            "features",
            "z_features",
        ]
        for key in candidates:
            if key in data:
                return key

        raise KeyError(f"Cannot find depth feature key. Tried {candidates}. Available keys: {list(data.keys())}")

    def _find_rgb_indices_key(self, data: Dict[str, Any]) -> Optional[str]:
        candidates = ["z_rgb_indices", "rgb_indices", "indices", "z_indices", "delta"]
        for key in candidates:
            if key in data:
                return key
        return None

    def _get_rgb_feature_sample(self, index: int) -> Dict[str, torch.Tensor]:
        part_idx, local_idx = self._global_to_local(index, self.rgb_feature_cum_counts)
        data = self._load_rgb_part(part_idx)

        feature_key = self._find_rgb_feature_key(data)
        z_rgb_features = data[feature_key][local_idx]

        if not torch.is_tensor(z_rgb_features):
            z_rgb_features = torch.tensor(z_rgb_features)

        out = {
            "z_rgb_features": z_rgb_features.float(),
        }

        if self.keep_z_rgb_indices:
            indices_key = self._find_rgb_indices_key(data)
            if indices_key is None:
                raise KeyError(
                    f"keep_z_rgb_indices=True but cannot find indices key. "
                    f"Available keys: {list(data.keys())}"
                )

            z_rgb_indices = data[indices_key][local_idx]
            if not torch.is_tensor(z_rgb_indices):
                z_rgb_indices = torch.tensor(z_rgb_indices)

            out["z_rgb_indices"] = z_rgb_indices.long()

        return out

    def _get_depth_feature_sample(self, index: int) -> Dict[str, torch.Tensor]:
        part_idx, local_idx = self._global_to_local(index, self.depth_feature_cum_counts)
        data = self._load_depth_feature_part(part_idx)

        feature_key = self._find_depth_feature_key(data)
        z_depth_feature = data[feature_key][local_idx]

        if not torch.is_tensor(z_depth_feature):
            z_depth_feature = torch.tensor(z_depth_feature)

        out = {
            "z_depth_feature": z_depth_feature.float(),
        }

        if self.check_id_alignment and "id" in data:
            out["id_from_depth_feature"] = str(data["id"][local_idx])

        return out

    def _load_depth(self, path: Union[str, Path]) -> torch.Tensor:
        """
        Load depth image using the same convention as Stage 1:
            uint16 depth -> float32 [0, 1] -> [1, H, W] -> resize -> repeat to 3ch
        """
        path = str(path)
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if depth is None:
            raise RuntimeError(f"Cannot read depth image: {path}")

        if depth.ndim != 2:
            raise RuntimeError(f"Depth image not single-channel: {path}, shape={depth.shape}")

        depth = depth.astype(np.float32) / self.depth_scale
        depth = np.clip(depth, 0.0, 1.0)

        depth = torch.from_numpy(depth).unsqueeze(0)
        depth = self.resize_depth(depth)

        if self.repeat_depth_to_3ch:
            depth = depth.repeat(3, 1, 1)

        return depth.float()

    def _get_first_existing_key(self, item: Dict[str, Any], keys: Sequence[str]) -> Any:
        for key in keys:
            if key in item:
                return item[key]
        raise KeyError(f"Missing all keys: {keys}. Available keys: {list(item.keys())}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.items[index]
        rgb_sample = self._get_rgb_feature_sample(index)
        depth_feature_sample = self._get_depth_feature_sample(index)

        if self.check_id_alignment:
            depth_feat_id = depth_feature_sample.get("id_from_depth_feature", None)
            if depth_feat_id is not None and str(item["id"]) != str(depth_feat_id):
                raise ValueError(
                    f"id mismatch at index={index}: jsonl id={item['id']} vs depth feature id={depth_feat_id}"
                )

        out = {
            "depth1": self._load_depth(item["depth1_path"]),
            "z_rgb_features": rgb_sample["z_rgb_features"],
            "z_depth_feature": depth_feature_sample["z_depth_feature"],
            "id": str(item["id"]),
            "depth1_path": str(item["depth1_path"]),
        }

        if self.keep_z_rgb_indices and "z_rgb_indices" in rgb_sample:
            out["z_rgb_indices"] = rgb_sample["z_rgb_indices"]

        return out


def build_stage252_dataset_model4(
    z_depth_path: Union[str, Path] = "/datasets/ssv2/nips/z_depth_train.jsonl",
    z_rgb_feature_manifest: Union[str, Path] = "/datasets/ssv2/nips/features/z_rgb_train_all_manifest.json",
    z_depth_feature_manifest: Union[str, Path] = "/datasets/ssv2/nips/features_depth_stage1/z_depth_train_stage1_manifest.json",
    image_size: Union[int, Sequence[int]] = 256,
    repeat_depth_to_3ch: bool = True,
    depth_scale: float = 65535.0,
    strict: bool = False,
    check_length_alignment: bool = True,
    rgb_feature_key: Optional[str] = None,
    depth_feature_key: Optional[str] = None,
    keep_z_rgb_indices: bool = False,
    check_id_alignment: bool = False,
) -> Stage252DatasetModel4:
    return Stage252DatasetModel4(
        z_depth_path=z_depth_path,
        z_rgb_feature_manifest=z_rgb_feature_manifest,
        z_depth_feature_manifest=z_depth_feature_manifest,
        image_size=image_size,
        repeat_depth_to_3ch=repeat_depth_to_3ch,
        depth_scale=depth_scale,
        strict=strict,
        check_length_alignment=check_length_alignment,
        rgb_feature_key=rgb_feature_key,
        depth_feature_key=depth_feature_key,
        keep_z_rgb_indices=keep_z_rgb_indices,
        check_id_alignment=check_id_alignment,
    )
