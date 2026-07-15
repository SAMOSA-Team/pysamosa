"""Legacy PurpleAir processing classes (pre-xarray workflow)."""

import numpy as np
import pandas as pd


def qa_bam(df_bam: pd.DataFrame, start_time=None) -> pd.DataFrame:
    """Apply quality assurance flags to a BAM DataFrame.

    Args:
        df_bam: BAM DataFrame with 'Status' and 'ConcHR(ug/m3)' columns.
        start_time: Optional start time to trim the DataFrame.

    Returns:
        QA'd BAM DataFrame with 'QA_flags' column added.
    """
    df_qa_bam = df_bam.copy()
    df_qa_bam.loc[:, "QA_flags"] = 0
    df_qa_bam.loc[df_qa_bam.Status != 0, "QA_flags"] = 1
    df_qa_bam.loc[
        (df_qa_bam.loc[:, "ConcHR(ug/m3)"] > 3)
        & (df_qa_bam.loc[:, "ConcHR(ug/m3)"] < 1000),
        "QA_flags",
    ] = 2
    if start_time:
        df_qa_bam = df_qa_bam.loc[start_time:, :]
    df_qa_bam = df_qa_bam.resample("1h").mean()
    df_qa_bam.loc[np.isnan(df_qa_bam.loc[:, "ConcHR(ug/m3)"]), "QA_flags"] = 3
    return df_qa_bam


class PurpleAir:
    """Legacy single-sensor PurpleAir data handler."""

    def __init__(
        self, name: str, df: pd.DataFrame, init_date: str = "df", threshold: float = 0.2
    ):
        """Args:
        name: Sensor name.
        df: DataFrame with 'a', 'b', 'rh' columns.
        init_date: Start date string, or 'df' to use first timestamp in data.
        threshold: Channel disagreement threshold for flagging.
        """
        self.name = name
        self.data = df.resample("1h").mean()
        self.init_date = df.index[0] if init_date == "df" else pd.Timestamp(init_date)
        self.threshold = threshold
        self.flag = pd.DataFrame(
            np.zeros((self.data.shape[0],)), index=self.data.index, columns=[self.name]
        ).astype(int)

    def diff_sensor(self) -> None:
        """Compute channel disagreement and store in self.data['disagree']."""
        self.data["disagree"] = (((self.data.a - self.data.b) ** 2) ** 0.5) / (
            (self.data.a + self.data.b) * 0.5
        )

    def flag_sensor(self) -> None:
        """Flag missing and high-disagreement observations in self.flag."""
        self.diff_sensor()
        down = self.data[self.data.isna().any(axis=1)].index
        ab_disagree = self.data[self.data.disagree > self.threshold].index
        self.flag.loc[down, self.name] = 1
        self.flag.loc[ab_disagree, self.name] = 2

    def uptime_report(self) -> pd.DataFrame:
        """Return a per-hour uptime report for this sensor.

        Returns:
            DataFrame with columns 'hour', 'flag', 'flag_count', 'flag_relative_count'.
        """
        self.flag_sensor()
        flag = self.flag
        flag_melt = flag.groupby(flag.index.hour).value_counts().reset_index()
        flag_melt.columns = ["hour", "flag", "flag_count"]
        flag_melt["flag_relative_count"] = (
            flag_melt["flag_count"] / flag_melt["flag_count"].sum()
        )
        return flag_melt

    def clean_sensor(self) -> pd.DataFrame:
        """Return a cleaned DataFrame with flagged observations removed.

        Returns:
            DataFrame with columns [sensor_name, 'rh'].
        """
        self.flag_sensor()
        data = self.data.loc[self.flag[self.flag == 0].dropna().index, :]
        mean = (data.a + data.b) * 0.5
        clean = pd.concat([mean, data.rh], axis=1).astype(float).round(2)
        clean.columns = [self.name, "rh"]
        return clean.resample("1h").mean()


class PurpleAirNetwork:
    """Legacy multi-sensor PurpleAir network handler."""

    def __init__(
        self,
        sensor_name_list: list[str],
        gdf,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        df_rh: pd.DataFrame,
        threshold: float = 0.2,
    ):
        """Args:
        sensor_name_list: List of sensor names.
        gdf: GeoDataFrame of sensor metadata.
        df_a: Channel A PM2.5 DataFrame (columns = sensor names).
        df_b: Channel B PM2.5 DataFrame (columns = sensor names).
        df_rh: Relative humidity DataFrame (columns = sensor names).
        threshold: Channel disagreement threshold.
        """
        self.data = {
            name: PurpleAir(
                name,
                pd.concat(
                    [df_a.loc[:, name], df_b.loc[:, name], df_rh.loc[:, name]], axis=1
                )
                .rename(columns={"a": "a", "b": "b", "rh": "rh"})
                .resample("1h")
                .mean(),
                threshold=threshold,
            )
            for name in sensor_name_list
        }
        self.meta = gdf
        self.sensor_name_list = sensor_name_list

    def diff_network(self) -> pd.DataFrame:
        """Return the per-sensor channel disagreement DataFrame.

        Returns:
            DataFrame of disagreement values.
        """
        return pd.concat(
            [self.data[i].data.loc[:, "disagree"] for i in self.data], axis=1
        ).rename(columns=dict(zip(self.data.keys(), self.sensor_name_list)))

    def clean_sensors(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return cleaned PM2.5 and RH DataFrames with flagged data removed.

        Returns:
            Tuple of (pm25_clean, rh_clean) DataFrames.
        """
        network_data = pd.DataFrame(columns=self.sensor_name_list).astype(float)
        network_rh = pd.DataFrame(columns=self.sensor_name_list).astype(float)
        for i in self.data:
            clean = self.data[i].clean_sensor()
            network_data.loc[:, i] = clean.iloc[:, 0]
            network_rh.loc[:, i] = clean.iloc[:, 1]
        return network_data, network_rh
