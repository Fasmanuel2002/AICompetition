import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation as R


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.scaler.set_output(transform="pandas")

    def _add_linear_acc(self, df: pl.DataFrame) -> pl.DataFrame:
        acc_values = df.select(["acc_x", "acc_y", "acc_z"]).to_numpy()
        quat_values = df.select(["rot_x", "rot_y", "rot_z", "rot_w"]).to_numpy()
        linear_acc_values = np.full_like(acc_values, np.nan)
        gravity_world = np.array([0, 0, 9.81])

        for i in range(len(df)):
            if np.all(np.isnan(quat_values[i])):
                continue
            rotation = R.from_quat(quat_values[i])
            gravity_values = rotation.apply(gravity_world, inverse=True)
            linear_acc_values[i, :] = acc_values[i, :] - gravity_values

        df_add = (
            pl.DataFrame(
                linear_acc_values,
                schema=["linear_acc_x", "linear_acc_y", "linear_acc_z"],
            )
            .fill_nan(None)
        )
        df = pl.concat([df, df_add], how="horizontal")
        return df

    def _add_rotvec_diff(self, df: pl.DataFrame) -> pl.DataFrame:
        ids = df.select("sequence_id").to_series().to_numpy()
        quat_values = df.select(["rot_x", "rot_y", "rot_z", "rot_w"]).to_numpy()
        rotvec_diff_values = np.full((len(df), 3), np.nan)

        for i in range(1, len(df)):
            if ids[i - 1] != ids[i]:
                continue
            q1 = quat_values[i - 1]
            q2 = quat_values[i]
            if np.all(np.isnan(q1)) or np.all(np.isnan(q2)):
                continue
            rot1 = R.from_quat(q1)
            rot2 = R.from_quat(q2)
            rotvec_diff_values[i, :] = (rot1.inv() * rot2).as_rotvec()

        df_add = (
            pl.DataFrame(
                rotvec_diff_values,
                schema=["rotvec_diff_x", "rotvec_diff_y", "rotvec_diff_z"],
            )
            .fill_nan(None)
        )
        df = pl.concat([df, df_add], how="horizontal")
        return df

    def _cancel_z_rotation(self, df: pl.DataFrame) -> pl.DataFrame:
        quat_values = df.select(["rot_x", "rot_y", "rot_z", "rot_w"]).to_numpy()
        rotate_flags = df.select("rotate").to_series().to_numpy()

        for i in range(len(df)):
            if np.all(np.isnan(quat_values[i])):
                continue
            rotation = (
                R.from_rotvec(
                    [0, 0, 130 + 180 * rotate_flags[i]],
                    degrees=True,
                )
                * R.from_quat(quat_values[i])
            )
            quat_values[i, :] = rotation.as_quat(
                canonical=True,
                scalar_first=True,
            )

        df_add = (
            pl.DataFrame(
                quat_values,
                schema=["rot_w", "rot_x", "rot_y", "rot_z"],
            )
            .fill_nan(None)
        )

        df = pl.concat(
            [df.drop(["rot_w", "rot_x", "rot_y", "rot_z"]), df_add],
            how="horizontal",
        )
        return df

    def _add_global_acc(self, df: pl.DataFrame) -> pl.DataFrame:
        acc_values = df.select(["acc_x", "acc_y", "acc_z"]).to_numpy()
        quat_values = df.select(["rot_x", "rot_y", "rot_z", "rot_w"]).to_numpy()
        global_acc_values = np.full_like(acc_values, np.nan)

        for i in range(len(df)):
            if np.all(np.isnan(quat_values[i])):
                continue
            rotation = R.from_quat(quat_values[i])
            global_acc_values[i, :] = rotation.apply(acc_values[i, :])

        df_add = (
            pl.DataFrame(
                global_acc_values,
                schema=["global_acc_x", "global_acc_y", "global_acc_z"],
            )
            .fill_nan(None)
        )

        df = pl.concat([df, df_add], how="horizontal")
        return df

    def _handle_left_handed(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df.with_columns(
                *[
                    pl.when(pl.col("handedness") == 0)
                    .then(-pl.col(col))
                    .otherwise(pl.col(col))
                    .alias(col)
                    for col in [
                        "acc_x",
                        "linear_acc_x",
                        "global_acc_x",
                        "rot_y",
                        "rot_z",
                        "rotvec_diff_y",
                        "rotvec_diff_z",
                    ]
                ],
            )
            .with_columns(
                pl.when(pl.col("handedness") == 0)
                .then(pl.col("thm_5"))
                .otherwise(pl.col("thm_3"))
                .alias("thm_3"),
                pl.when(pl.col("handedness") == 0)
                .then(pl.col("thm_3"))
                .otherwise(pl.col("thm_5"))
                .alias("thm_5"),
                *[
                    pl.when(pl.col("handedness") == 0)
                    .then(pl.col(f"tof_5_v{i}"))
                    .otherwise(pl.col(f"tof_3_v{i}"))
                    .alias(f"tof_3_v{i}")
                    for i in range(64)
                ],
                *[
                    pl.when(pl.col("handedness") == 0)
                    .then(pl.col(f"tof_3_v{i}"))
                    .otherwise(pl.col(f"tof_5_v{i}"))
                    .alias(f"tof_5_v{i}")
                    for i in range(64)
                ],
            )
            .with_columns(
                *[
                    pl.when(pl.col("handedness") == 0)
                    .then(pl.col(f"tof_{i}_v{8 * j + 7 - k}"))
                    .otherwise(pl.col(f"tof_{i}_v{8 * j + k}"))
                    .alias(f"tof_{i}_v{8 * j + k}")
                    for i in range(1, 6)
                    for j in range(8)
                    for k in range(8)
                ],
            )
        )

        return df

    def _handle_rotated_device(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df.with_columns(
                *[
                    pl.when(pl.col("rotate") == 1)
                    .then(-pl.col(col))
                    .otherwise(pl.col(col))
                    .alias(col)
                    for col in [
                        "acc_x",
                        "acc_y",
                        "linear_acc_x",
                        "linear_acc_y",
                        "global_acc_x",
                        "global_acc_y",
                        "rot_x",
                        "rot_y",
                        "rotvec_diff_x",
                        "rotvec_diff_y",
                    ]
                ],
            )
            .with_columns(
                pl.when(pl.col("rotate") == 1)
                .then(pl.col("thm_4"))
                .otherwise(pl.col("thm_2"))
                .alias("thm_2"),
                pl.when(pl.col("rotate") == 1)
                .then(pl.col("thm_2"))
                .otherwise(pl.col("thm_4"))
                .alias("thm_4"),
                pl.when(pl.col("rotate") == 1)
                .then(pl.col("thm_5"))
                .otherwise(pl.col("thm_3"))
                .alias("thm_3"),
                pl.when(pl.col("rotate") == 1)
                .then(pl.col("thm_3"))
                .otherwise(pl.col("thm_5"))
                .alias("thm_5"),
                *[
                    pl.when(pl.col("rotate") == 1)
                    .then(pl.col(f"tof_4_v{i}"))
                    .otherwise(pl.col(f"tof_2_v{i}"))
                    .alias(f"tof_2_v{i}")
                    for i in range(64)
                ],
                *[
                    pl.when(pl.col("rotate") == 1)
                    .then(pl.col(f"tof_2_v{i}"))
                    .otherwise(pl.col(f"tof_4_v{i}"))
                    .alias(f"tof_4_v{i}")
                    for i in range(64)
                ],
                *[
                    pl.when(pl.col("rotate") == 1)
                    .then(pl.col(f"tof_5_v{i}"))
                    .otherwise(pl.col(f"tof_3_v{i}"))
                    .alias(f"tof_3_v{i}")
                    for i in range(64)
                ],
                *[
                    pl.when(pl.col("rotate") == 1)
                    .then(pl.col(f"tof_3_v{i}"))
                    .otherwise(pl.col(f"tof_5_v{i}"))
                    .alias(f"tof_5_v{i}")
                    for i in range(64)
                ],
            )
            .with_columns(
                *[
                    pl.when(pl.col("rotate") == 1)
                    .then(pl.col(f"tof_{i}_v{63 - j}"))
                    .otherwise(pl.col(f"tof_{i}_v{j}"))
                    .alias(f"tof_{i}_v{j}")
                    for i in range(1, 6)
                    for j in range(64)
                ],
            )
        )

        return df

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        if "rotate" not in df.columns:
            df = df.with_columns(pl.lit(0).alias("rotate"))

        df = self._add_linear_acc(df)
        df = self._add_rotvec_diff(df)
        df = self._cancel_z_rotation(df)
        df = self._add_global_acc(df)
        df = self._handle_left_handed(df)
        df = self._handle_rotated_device(df)

        
        df = df.with_columns(
            pl.col("^tof_._v.*$").replace({-1: 255}),
        )

        return df

    def _get_reversed_sequence_tail(
        self,
        df: pl.DataFrame,
        seq_len: int,
    ) -> pl.DataFrame:
        df_tail = (
            df.sort(
                ["sequence_id", "sequence_counter"],
                descending=[False, True],
            )
            .group_by("sequence_id", maintain_order=True)
            .head(seq_len)
            .with_columns(
                pl.col("sequence_counter")
                .cum_count()
                .over("sequence_id"),
            )
        )

        return df_tail

    def get_feature_array(
        self,
        df: pl.DataFrame,
        list_features: list,
        seq_len: int,
        fit: bool = False,
    ) -> np.ndarray:
        df_tail = self._get_reversed_sequence_tail(df, seq_len)

        if fit:
            self.scaler.fit(df_tail.select(list_features))

        df_tail = df_tail.to_pandas()
        df_tail[list_features] = self.scaler.transform(
            df_tail[list_features]
        )
        df_tail = pl.from_pandas(df_tail).fill_null(0.0)

        feature_arrays = []
        for _, df_group in df_tail.group_by(
            "sequence_id",
            maintain_order=True,
        ):
            array = df_group.select(list_features).to_numpy().T
            array = np.pad(
                array,
                ((0, 0), (0, seq_len - array.shape[-1])),
            )
            feature_arrays.append(array)

        return np.stack(feature_arrays, axis=0)