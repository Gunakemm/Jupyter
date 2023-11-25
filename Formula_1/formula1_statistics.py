import pandas as pd
import numpy as np
import seaborn as sns
import os
from os.path import isfile, join
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage,AnnotationBbox
import psycopg2
from sqlalchemy import create_engine, text
import requests
from bs4 import BeautifulSoup
import re

db_params = {
    'host': 'localhost:5432',
    'database': 'Formula1',
    'user': 'postgres',
    'password': 'Gunakemm12814679'
}
engine = create_engine(f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}/{db_params["database"]}')

class YearStatistics:

    def __init__(self, year):
        self.year = year

    def __replace_with_previous_plus_1(self, series):
        result = []
        prev_value = None
        for value in series:
            if value == "\\N":
                if prev_value is not None:
                    result.append(prev_value + 1)
                    prev_value = int(prev_value + 1)
                else:
                    result.append(None)
            else:
                result.append(value)
                prev_value = int(value)
        return result

    # Box plot drivers results
    def drivers_championship(self):
        request = f"""
            WITH
            a2 AS
                (SELECT  driverid, forename || ' ' || surname AS fullname
                FROM drivers),
            a3 AS
                (SELECT * FROM colors),
            a4 AS
                (SELECT driverid, points, raceid
                 FROM sprint_results)

            SELECT a1.driverid, a2.fullname, a3.colorcode, coalesce(SUM(a1.points),0) + coalesce(SUM(a4.points),0) as points
            FROM results a1
            LEFT JOIN a2
                ON (a1.driverid = a2.driverid)
            LEFT JOIN a3
                ON (a1.constructorid = a3.constructorid)
            LEFT JOIN a4
                ON (a1.raceId = a4.raceId AND a1.driverId = a4.driverId)
            WHERE a1.raceId in
                (SELECT raceId
                 FROM races
                 WHERE year = {self.year}
                 ORDER BY raceId)
            GROUP BY a1.driverid, a1.constructorid, a2.fullname, a3.colorcode
            ORDER BY coalesce(SUM(a1.points),0) + coalesce(SUM(a4.points), 0) DESC;"""
        df = pd.read_sql_query(request, con=engine)

        sns.set(rc={'figure.figsize': (15, 12), 'axes.facecolor': 'white'})

        ax = sns.barplot(x=df['points'].to_numpy(),
                         y=df['fullname'].to_numpy(),
                         palette=df['colorcode'].to_numpy())
        ax.bar_label(ax.containers[0], fontsize=13)
        ax.set_title('Drivers championship ' + str(self.year), fontsize=24)

        ax.set_ylabel('Drivers', fontsize=20)
        ax.set_xlabel('Points', fontsize=20)

        plt.yticks(fontsize=16)
        plt.show()

    # Box plot constructors results
    def constructors_championship(self):
        request = f"""
        WITH
        a2 AS
            (SELECT * FROM colors),
        a3 AS
            (SELECT constructorid, name FROM constructors),
        a4 AS
            (SELECT raceid, points, constructorid, driverid FROM sprint_results)
        
        SELECT a2.colorcode, a3.name, coalesce(SUM(a1.points),0) + coalesce(SUM(a4.points),0) as points
        FROM results a1
        LEFT JOIN a2
            ON (a1.constructorid = a2.constructorid)
        LEFT JOIN a3
            ON (a1.constructorid = a3.constructorid)
        LEFT JOIN a4
            ON (a1.raceid = a4.raceid AND a1.driverid = a4.driverid)
        WHERE a1.raceId in
            (SELECT raceId
             FROM races
             WHERE year = {self.year}
             ORDER BY raceId)
        GROUP BY a2.colorcode, a3.name
        ORDER BY coalesce(SUM(a1.points),0) + coalesce(SUM(a4.points),0) DESC;"""
        df = pd.read_sql_query(request, con=engine)
        sns.set(rc={'figure.figsize': (15, 12), 'axes.facecolor': 'white'})

        ax = sns.barplot(x=df['points'].to_numpy(),
                         y=df['name'].to_numpy(),
                         palette=df['colorcode'].to_numpy())
        ax.bar_label(ax.containers[0], fontsize=13)
        ax.set_title('Constructors championship ' + str(self.year), fontsize=24)

        ax.set_ylabel('Constructor', fontsize=20)
        ax.set_xlabel('Points', fontsize=20)

        plt.yticks(fontsize=16)
        plt.show()

    # Line plot drivers season
    def drivers_points_during_season(self):
        query = f"""
        WITH
        a2 AS
            (SELECT * FROM colors),
        a3 AS
            (SELECT driverid, surname FROM drivers),
        a4 AS
            (SELECT raceid, name FROM races)

        SELECT a1.driverid, a1.points, a2.colorcode, a3.surname, a4.name as track_name
        FROM results a1
        JOIN a2 ON a1.constructorid = a2.constructorid
        JOIN a3 ON a1.driverid = a3.driverid
        JOIN a4 ON a1.raceid = a4.raceid
        WHERE a1.raceid in
            (SELECT raceId
             FROM races
             WHERE year = {self.year})
        ORDER BY a1.raceid, a1.driverid;"""
        df = pd.read_sql_query(query, con=engine)
        df['cumulative_points'] = df.groupby('driverid')['points'].cumsum()
        drivers_info = df.groupby('driverid')[['colorcode', 'surname']].first()
        color_dict = drivers_info['colorcode'].to_dict()

        sns.set_style("whitegrid")
        fig, ax = plt.subplots()

        sns.lineplot(df,
                     x='track_name',
                     y='cumulative_points',
                     hue='driverid',
                     palette=color_dict,
                     ax=ax,
                     legend=None)
        ax.set_xticks(range(len(df['track_name'].unique())), list(df['track_name'].unique()), rotation='vertical')
        for line, name in zip(ax.lines, drivers_info['surname'].to_numpy()):
            y = line.get_ydata()[-1]
            x = line.get_xdata()[-1]
            if not np.isfinite(y):
                y = next(reversed(line.get_ydata()[~line.get_ydata().mask]), float("nan"))
            if not np.isfinite(y) or not np.isfinite(x):
                continue
            text = ax.annotate(name,
                               xy=(x, y),
                               xytext=(0, 0),
                               color=line.get_color(),
                               xycoords=(ax.get_xaxis_transform(),
                                         ax.get_yaxis_transform()),
                               textcoords="offset points")
            text_width = (text.get_window_extent(
                fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width)
            # if np.isfinite(text_width):
            #     ax.set_xlim(ax.get_xlim()[0], text.xy[0] + text_width * 0.90)

        ax.set_title('Drivers championship ' + str(self.year), fontsize=24)
        ax.set_ylabel('Points', fontsize=20)
        ax.set_xlabel('')
        plt.show()

    # Line plot constructors season
    def constructors_points_during_season(self):
        query = f"""
        WITH
        a2 AS
            (SELECT * FROM colors),
        a3 AS
            (SELECT constructorid, name FROM constructors),
        a4 AS
            (SELECT raceid, name FROM races)

        SELECT a1.constructorid, a4.name as track_name, a2.colorcode, a3.name, SUM(a1.points) AS points
        FROM results a1
        JOIN a2 ON a1.constructorid = a2.constructorid
        JOIN a3 ON a1.constructorid = a3.constructorid
        JOIN a4 ON a1.raceid = a4.raceid
        WHERE a1.raceid in
            (SELECT raceId
             FROM races
             WHERE year = {self.year})
        GROUP BY a1.constructorid, a1.raceid, a4.name, a2.colorcode, a3.name
        ORDER BY a1.raceid, a1.constructorid;
        """
        df = pd.read_sql_query(query, con=engine)
        df['cumulative_points'] = df.groupby('constructorid')['points'].cumsum()
        constructors_info = df.groupby('constructorid')[['colorcode', 'name']].first()
        color_dict = constructors_info['colorcode'].to_dict()

        sns.set_style("whitegrid")
        fig, ax = plt.subplots()

        sns.lineplot(df,
                     x='track_name',
                     y='cumulative_points',
                     hue='constructorid',
                     palette=color_dict,
                     ax=ax,
                     legend=None)
        ax.set_xticks(range(len(df['track_name'].unique())), list(df['track_name'].unique()), rotation='vertical')
        for line, name in zip(ax.lines, constructors_info['name'].to_numpy()):
            y = line.get_ydata()[-1]
            x = line.get_xdata()[-1]
            if not np.isfinite(y):
                y = next(reversed(line.get_ydata()[~line.get_ydata().mask]), float("nan"))
            if not np.isfinite(y) or not np.isfinite(x):
                continue
            text = ax.annotate(name,
                               xy=(x, y),
                               xytext=(0, 0),
                               color=line.get_color(),
                               xycoords=(ax.get_xaxis_transform(),
                                         ax.get_yaxis_transform()),
                               textcoords="offset points")
            text_width = (text.get_window_extent(
                fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width)

    # Average start position for race
    def mean_start_position(self):
        query  = f"""
        WITH
        a2 AS
            (SELECT driverid, forename || ' ' || surname AS fullname
             FROM drivers)
        
        SELECT a2.fullname, round(AVG(a1.grid), 2) AS mean_start_position
        FROM results a1
        JOIN a2 ON a1.driverid = a2.driverid
        WHERE a1.raceid in
            (SELECT raceid
             FROM races
             WHERE year = {self.year})
        GROUP BY a2.fullname
        ORDER BY AVG(a1.grid)
        """
        return pd.read_sql_query(query, con=engine).set_index('fullname')

    # Median start position for race
    def median_start_position(self):
        query = f"""
        WITH
        a2 AS
            (SELECT driverid, forename || ' ' || surname AS fullname
             FROM drivers)
        
        SELECT a2.fullname, a1.grid
        FROM results a1
        JOIN a2 ON a1.driverid = a2.driverid
        WHERE a1.raceid in
            (SELECT raceid
             FROM races
             WHERE year = {self.year})
        """
        df = pd.read_sql_query(query, con=engine)
        return pd.DataFrame(df.groupby('fullname')['grid'].median()).sort_values('grid')

    # Average points for race
    def mean_points(self):
        query = f"""
        WITH
        a2 AS
            (SELECT driverid, forename || ' ' || surname AS fullname
             FROM drivers)
        
        SELECT a2.fullname, round(CAST(AVG(a1.points) as numeric), 2) as mean_points
        FROM results a1
        JOIN a2 ON a1.driverid = a2.driverid
        WHERE a1.raceid in
            (SELECT raceid
             FROM races
             WHERE year = 2021)
        GROUP BY a2.fullname
        ORDER BY AVG(a1.points) DESC;
        """
        return pd.DataFrame(pd.read_sql_query(query, con=engine))

    def percent_of_finishes(self):
        query = f"""
        WITH
        a2 AS
            (SELECT *
             FROM status)
        
        SELECT *
        FROM results a1
        JOIN a2 ON a1.statusid = a2.statusid
        WHERE raceid in
            (SELECT raceid
             FROM races
             WHERE year = {self.year})"""
        df = pd.read_sql_query(query, con=engine)
        return f'{np.round(df[(df["status"] == "Finished") | (df["status"].str.startswith("+"))]["status"].count() / df["status"].count() * 100, 2)} %'

    def start_finish_correlation(self):
        query = f"""
        WITH
        a2 AS
            (SELECT *
             FROM colors),
        a3 AS
            (SELECT constructorid, name
             FROM constructors)
        
        SELECT a1.grid as start_position, a1.position as finish_position, a3.name, a1.number, a2.colorcode
        FROM results a1
        JOIN a2 ON a1.constructorid = a2.constructorid
        JOIN a3 ON a1.constructorid = a3.constructorid
        WHERE raceid in
            (SELECT raceid
             FROM races
             WHERE year = {self.year})
        """
        df = pd.read_sql_query(query, con=engine)
        df['finish_position'] = self.__replace_with_previous_plus_1(df['finish_position'])
        df['finish_position'] = df['finish_position'].astype('int')
        df.drop(df[df['start_position'] == 0].index, inplace=True)

        fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(8, 14))

        color_dict = df.groupby('name')['colorcode'].first().to_dict()


        sns.scatterplot(df,
                        x='start_position',
                        y='finish_position',
                        hue='name',
                        palette=color_dict,
                        ax=axs[0]
                        )
        # x = df['start_position'].to_numpy()[::-1]
        # y = df['finish_position'].to_numpy()[::-1]
        # numbers = df['number'].to_numpy()[::-1]
        #
        # text_placed = {}
        # for i, (x_val, y_val) in enumerate(zip(x, y)):
        #     if str(x_val) + str(y_val) not in text_placed:
        #         axs[0].text(x_val, y_val, str(numbers[i]), fontsize=8, ha='center', va='bottom')
        #         text_placed[str(x_val) + str(y_val)] = 1

        axs[0].set_xlabel('Start position')
        axs[0].set_ylabel('Finish position')

        legend = axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

        correlation_matrix = df[['start_position', 'finish_position']].corr()
        correlation_matrix.style.background_gradient(cmap='coolwarm')
        sns.heatmap(correlation_matrix,
                    cmap='coolwarm',
                    annot=True,
                    ax=axs[1])
        plt.show()


class DriverStatistics:

    def __init__(self, name):
        self.name = name


    def __get_track_distance_from_wiki(self, url):
        response = requests.get(
            url=url,
        )
        soup = BeautifulSoup(response.content, 'html.parser')

        res = soup.find(id="bodyContent").find_all(string=re.compile(" km "))
        pattern = r'([\d.]+) km'

        km_values = []
        for item in res:
            match = re.search(pattern, item)
            if match:
                km_values.append(float(match.group(1)))

        return min(km_values)

    def podiums_results(self):
        query = f"""
        WITH
        a2 AS
            (SELECT constructorid, name
             FROM constructors)
        
        SELECT CASE WHEN a2.name IS NULL THEN 'total' ELSE a2.name END AS team_name,
               CASE WHEN a1.position IS NULL THEN 'total' ELSE a1.position END AS podium_position,
               COUNT(*)
        FROM results a1
        JOIN a2 ON a1.constructorid = a2.constructorid
        WHERE driverid =
              (SELECT driverid
               FROM drivers
               WHERE concat(forename,' ',surname) = '{self.name}')
                AND a1.position in ('1', '2', '3')
        GROUP BY ROLLUP(a2.name, a1.position)
        ORDER BY team_name, COUNT(*) DESC;"""
        df = pd.read_sql_query(query, con=engine)
        print(f'{self.name} podiums statistics:')
        return df

    def pace_in_race(self, year, track_name):
        pass

if __name__ ==  "__main__":
    print(123)