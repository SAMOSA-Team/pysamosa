def qa_bam(df_bam, start_time=None):
    """Performs quality assurance on a BAM DataFrame.

    Args:
        df_bam: A Pandas DataFrame containing BAM data.
        start_time: A datetime object specifying the start time for QA.

    Returns:
        A Pandas DataFrame containing the QA'd BAM data with flags for removed data.
    """

    df_qa_bam = df_bam.copy()
    df_qa_bam.loc[:, 'QA_flags'] = 0

    # Flags all self-reported bad datapoints
    df_qa_bam.loc[df_qa_bam.Status != 0, 'QA_flags'] = 1

    # Flags limits of detection from 3 - 1000 ug/m3 on an hourly basis
    # These bounds are reasonable for most applications
    df_qa_bam.loc[(df_qa_bam.loc['ConcHR(ug/m3)'] > 3) & (df_qa_bam.loc[:, 'ConcHR(ug/m3)'] < 1000), 'QA_flags'] = 2

    # Sets period beginning of campaign, if relevant
    if start_time:
        df_qa_bam = df_qa_bam.loc[start_time:, :]

    # Flag all off-times
    df_qa_bam = df_qa_bam.resample('1H').mean()
    df_qa_bam.loc[np.isnan(df_qa_bam.loc[:, 'ConcHR(ug/m3)']), 'QA_flags'] = 3

    return df_qa_bam

# Define Classes
class PurpleAir:
    """A class for representing and processing PurpleAir data.

    Attributes:
        name: The name of the PurpleAir sensor.
        data: A Pandas DataFrame containing the PurpleAir data.
        init_date: The start date of the PurpleAir data.
        threshold: The threshold for flagging data.
        flag: A Pandas DataFrame containing the flags for the PurpleAir data.
    """

    def __init__(self, name, df, init_date='df', threshold=0.2):
        """Initializes a new PurpleAir object.

        Args:
            name: The name of the PurpleAir sensor.
            df: A Pandas DataFrame containing the PurpleAir data.
            init_date: The start date of the PurpleAir data (defaults to the first date in the DataFrame).
            threshold: The threshold for flagging data (defaults to 0.2).
        """

        self.name = name
        self.data = df.resample('1H').mean()
        if init_date == 'df':
            self.init_date = df.index[0]
        else:
            self.init_date = pd.Timestamp(init_date)
        self.threshold = threshold
        self.flag = pd.DataFrame(np.zeros((self.data.shape[0],)), index=self.data.index, columns=[self.name]).astype(
            int)

    def diff_sensor(self):
        """Calculates the disagreement between the two sensors of the PurpleAir device.

        Stores the disagreement in the `disagree` column of the `data` DataFrame.
        """

        self.data['disagree'] = (((self.data.a - self.data.b) ** 2) ** 0.5) / ((self.data.a + self.data.b) * 0.5)

    def flag_sensor(self):
        """Flags data that is missing or has a disagreement above the threshold.

        Stores the flags in the `flag` DataFrame.
        """

        self.diff_sensor()
        down = self.data[self.data.isna().any(axis=1)].index
        ab_disagree = self.data[self.data.disagree > self.threshold].index
        self.flag.loc[down, self.name] = 1
        self.flag.loc[ab_disagree, self.name] = 2

    def uptime_report(self):
        """Generates a report of the uptime of the PurpleAir device.

        The report is a Pandas DataFrame showing the number and fraction of flags for each hour of the day.

        Returns:
            A Pandas DataFrame containing the uptime report.
        """

        self.flag_sensor()
        flag = self.flag
        flag_melt = flag.groupby(flag.index.hour).value_counts().reset_index()
        flag_melt.columns = ['hour', 'flag', 'flag_count']
        flag_melt['flag_relative_count'] = flag_melt['flag_count'] / flag_melt['flag_count'].sum()
        return flag_melt

    def clean_sensor(self):
        """Cleans the PurpleAir data by removing flagged data.

        Returns:
            A Pandas DataFrame containing the cleaned PurpleAir data.
        """

        self.flag_sensor()
        data = self.data.loc[self.flag[self.flag == 0].dropna().index, :]
        mean = (data.a + data.b) * 0.5
        clean = pd.concat([mean, data.rh], axis=1).astype(float).round(2)
        clean.columns = [self.name, 'rh']
        clean = clean.resample('1H').mean()
        return clean


class PurpleAirNetwork:
    """A class for representing and processing a network of PurpleAir sensors.

    Attributes:
        data: A dictionary mapping sensor names to `PurpleAir` objects.
        meta: A GeoPandas GeoDataFrame containing metadata for the PurpleAir sensors.
        sensor_name_list: A list of sensor names.
    """

    def __init__(self, sensor_name_list, gdf, df_a, df_b, df_rh, threshold=0.2):
        """Initializes a new PurpleAirNetwork object.

        Args:
            sensor_name_list: A list of sensor names.
            gdf: A GeoPandas GeoDataFrame containing metadata for the PurpleAir sensors.
            df_a: A Pandas DataFrame containing the data from sensor A.
            df_b: A Pandas DataFrame containing the data from sensor B.
            df_rh: A Pandas DataFrame containing the relative humidity data.
            threshold: The threshold for flagging data (defaults to 0.2).
        """

        self.data = {
            sensor_name: PurpleAir(sensor_name, pd.concat(
                [df_a.loc[:, sensor_name], df_b.loc[:, sensor_name], df_rh.loc[:, sensor_name]], axis=1).rename(
                columns={'a': 'a', 'b': 'b', 'rh': 'rh'}).resample('1H').mean(), threshold=threshold)
            for sensor_name in sensor_name_list
        }
        self.meta = gdf
        self.sensor_name_list = sensor_name_list

    def diff_network(self):
        """Calculates the disagreement between the two sensors of each PurpleAir device in the network.

        Returns:
            A Pandas DataFrame containing the disagreement for each sensor.
        """

        return pd.concat([self.data[i].data.loc[:, 'disagree'] for i in self.data.keys()], axis=1).rename(
            columns=self.sensor_name_list)

    def clean_sensors(self):
        """Cleans the PurpleAir data by removing flagged data.

        Returns:
            A tuple containing two Pandas DataFrames:
                * The first DataFrame contains the cleaned PM2.5 data.
                * The second DataFrame contains the cleaned relative humidity data.
        """

        network_data = pd.DataFrame(columns=self.sensor_name_list).astype(float)
        network_rh = pd.DataFrame(columns=self.sensor_name_list).astype(float)
        for i in self.data.keys():
            clean = self.data[i].clean_sensor()
            network_data.loc[:, i] = clean.iloc[:, 0]
            network_rh.loc[:, i] = clean.iloc[:, 1]
        return network_data, network_rh

    def settlement_agreement(self):
        """Flags data from sensors that are in settlements where there is insufficient data or where the sensor data
        is too different from the mean settlement value.

        This method updates the `flag` attribute of each `PurpleAir` object in the network.
        """

        self.clean_sensors()
        for i in self.sensor_name_list:
            # at some point meta.site_id should be replaced with "sensor_name" or vice-versa
            settlement_sensors = self.meta.loc[self.meta.distance(
                self.meta.loc[self.meta[self.meta.site_id == i].index[0], :].geometry) <= 10000, 'site_id'].values
            settlement_data = self.network_data.loc[:, settlement_sensors].dropna(axis=0, how='all')
            settlement_data = settlement_data.groupby(settlement_data.columns, axis=1).mean()

            # Flag insufficient data
            settlement_insufficient = settlement_data[settlement_data.count(axis=1) < 3].index
            self.data[i].flag.loc[settlement_insufficient] = 3

            # Flag data that is too different from the mean settlement value
            settlement_data = settlement_data.drop(settlement_insufficient, axis=0)
            settlement_data.loc[:, 'settlement_mean'] = settlement_data.mean(axis=1)
            settlement_baseline = np.abs(
                settlement_data.loc[:, i].subtract(settlement_data.loc[:, 'settlement_mean'])).divide(
                settlement_data.loc[:, 'settlement_mean'])
            settlement_diff = np.where(settlement_baseline > 1.2, self.data[i].flag == 0, self.data[i].flag == 4)
            self.data[i].flag.loc[settlement_diff] = 4

    def flag_network(self):
        """Returns a DataFrame of flags for each sensor in the network.

        The flags indicate whether the sensor data is valid and reliable.
        """

        self.settlement_agreement()
        network_flag = pd.DataFrame(columns=self.sensor_name_list).astype(int)
        for i in self.sensor_name_list:
            network_flag.loc[:, i] = self.data[i].flag
        return network_flag

    def clean_settlement(self):
        """Returns a DataFrame of the mean PM2.5 concentration for each settlement in the network.

        The data is cleaned by removing flagged data.
        """

        network_flag = self.flag_network()
        settlements = self.meta.settlement_name.unique()
        settlements_data = pd.DataFrame(columns=settlements).astype(float)
        for i in settlements:
            sites = self.meta.loc[self.meta.settlement_name == i, :].site_id
            settlement_flag = network_flag.loc[:, sites]
            settlement_data = self.network_data[network_flag == 0].loc[:, sites]
            settlements_data.loc[:, i] = settlement_data.mean(axis=1)
        return settlements_data

    def clean_settlement_type(self):
        """Returns a DataFrame of the mean PM2.5 concentration for each settlement type in the network.

        The data is cleaned by removing flagged data.
        """

        network_flag = self.flag_network()
        settlement_type = self.meta.settlement_type.unique()
        settlement_type_data = pd.DataFrame(columns=settlement_type).astype(float)
        for i in settlement_type:
            sites = self.meta.loc[self.meta.settlement_type == i, :].site_id
            settlement_flag = network_flag.loc[:, sites]
            settlement_data = self.network_data[network_flag == 0].loc[:, sites]
            settlement_type_data.loc[:, i] = settlement_data.mean(axis=1)
        return settlement_type_data

    def clean_cluster(self):
        """Returns a DataFrame of the mean PM2.5 concentration for each cluster in the network.

        The data is cleaned by removing flagged data.
        """

        network_flag = self.flag_network()
        cluster_names = self.meta.cluster_name.unique()
        cluster_data = pd.DataFrame(columns=cluster_names).astype(float)
        for cluster_name in cluster_names:
            sites = self.meta.loc[self.meta.cluster_name == cluster_name, :].site_id
            settlement_flag = network_flag.loc[:, sites]
            settlement_data = self.network_data[network_flag == 0].loc[:, sites]
            cluster_data.loc[:, cluster_name] = settlement_data.mean(axis=1)
        return cluster_data


def model_bayes(df, sigma=[1., 1., 1.], form='ols'):
    if form == 'ols':
        model_randomwalk = pm.Model()
        with model_randomwalk:

            sigma_cf1 = pm.Exponential("sigma_cf1", sigma[0])
            beta_cf1 = pm.GaussianRandomWalk("beta_cf1", sigma=sigma_cf1, shape=len(df.BAM))

            sigma_rh = pm.Exponential("sigma_rh", sigma[1])
            beta_rh = pm.GaussianRandomWalk("beta_rh", sigma=sigma_rh, shape=len(df.BAM))

            sigma_intercept = pm.Exponential("sigma_intercept", sigma[2])
            beta_intercept = pm.GaussianRandomWalk("beta_intercept", sigma=sigma_intercept, shape=len(df.BAM))

        with model_randomwalk:
            regression = df.CF1 * beta_cf1 + df.RH * beta_rh + beta_intercept
            sd = pm.HalfNormal("sd", sigma=1)
            likelihood = pm.Normal("y", mu=regression, sigma=sd, observed=df.BAM)

        with model_randomwalk:
            trace_rw = pm.sample(tune=2000, cores=10, target_accept=0.9)

        y_hat = (df.CF1 * trace_rw['beta_cf1'] + df.RH * trace_rw['beta_rh'] + trace_rw['beta_intercept']).median()

    elif form == 'nls':
        model_randomwalk = pm.Model()
        with model_randomwalk:

            sigma_rh = pm.Exponential("sigma_rh", sigma[1])
            beta_rh = pm.GaussianRandomWalk("beta_rh", sigma=sigma_rh, shape=len(df.BAM))

        with model_randomwalk:
            regression = df.CF1 / (1 + (beta_rh / (100 / df.RH - 1)))
            sd = pm.HalfNormal("sd", sigma=1)
            likelihood = pm.Normal("y", mu=regression, sigma=sd, observed=df.BAM)

        with model_randomwalk:
            trace_rw = pm.sample(tune=2000, cores=10, target_accept=0.9)

        k = (1 + (trace_rw['beta_rh'] / (100 / df.RH - 1))).median()
        y_hat = df.CF1 / k

    y_true = df.BAM
    metrics = {'R2': adj_r2(y_true, y_hat, 2),
               'RMSE': rmse(y_true, y_hat),
               'NRMSE': nrmse(y_true, y_hat),
               'MBE': mbe(y_true, y_hat),
               'NMBE': nmbe(y_true, y_hat)}

    return y_hat, metrics, trace_rw


df_resids.replace('IGP-CARE', 'Regional \n Background', inplace=True)
df_resids.replace('R K Puram', 'Urban \n High-Traffic 1', inplace=True)
df_resids.replace('Aurobindo Marg', 'Urban \n Background 1', inplace=True)
df_resids.replace('Najafgarh', 'Urban \n Background 2', inplace=True)
df_resids.replace('Patparganj', 'Urban \n High-Traffic 2', inplace=True)
df_resids.replace('Wazirpur', 'Urban \n Low-Traffic 1', inplace=True)
df_resids.replace('Punjabi Bagh', 'Urban \n Low-Traffic 2', inplace=True)
df_resids.replace('Mandir Marg', 'Urban \n Greenspace', inplace=True)
df_resids.replace('Vivek Vihar', 'Urban \n Residential', inplace=True)
df_resids.replace('IIT Delhi', 'Urban \n University', inplace=True)