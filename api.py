from flask import *
from flask_sqlalchemy import SQLAlchemy
from models import *
from charts import *
import io, base64, os
import random
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sqlalchemy import create_engine, func, and_, desc, asc, extract, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from matplotlib.figure import Figure
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

'''
Contains all functions used/called in emp_analysis_project.py (app file)
'''
PATH = os.getcwd()+"/"
os.chdir(PATH+"/data")
app = Flask(__name__)
app.config["sqlite:///health_care_portal.db"] = 'postgresql+psycopg2://login:pass@localhost/flask_app'
db = SQLAlchemy(app)
db.create_all()
db.session.commit()

Base = declarative_base()

engine = create_engine("sqlite:///health_care_portal.db?check_same_thread=False")
Base.metadata.bind = engine
Base.metadata.create_all(engine)
DBSession = sessionmaker(bind=engine)
session = DBSession()

class EmpAnalysis:

    def __init__(self, path):
        '''
        Connects to the directory where the files with raw data reside
        '''
        self.path = path+"/data"
        self.description = ProjectDescription.project_description(self.path+'/salary_analysis_description.txt')
        self.df = self.emp_list_df()
        self.genders = self.df.gender.unique()
        self.sg = self.df.s_g.unique()
        self.dept = self.df.dept.unique()
        self.dept.sort()
        self.emp_counts_by_dept_chart_barh = self.emp_counts_by_dept_page()
        self.dept_name = self.dept_dict()
        self.emp_active = self.emp_active_df()
        self.employee_counts = self.emp_count_page()
        self.active_emp_page = self.active_emp()
        self.df_emp_id = self.emp_id_df()
        self.df_raise = self.raise_schedule_df()
        self.salary_promotion = self.salary_promo_page()
        self.emp_salary_scatter_plots = self.scatter_plot_dict_for_emp_salary()

    def data(self):
        '''
        Import and Clean the Employee file
        '''
        df = pd.read_csv(self.path+"/emp_file_CAPSTONE.txt", names=["first_name", "last_name", "middle_initial", "gender", "s_g", "salary", "dept", "hire_date", "term_date"], index_col=False)
        df.fillna("", inplace=True)
        df.drop([0], inplace=True)
        for col in df.columns:
            if col == "hire_date" or col == "term_date":
                df[col] = pd.to_datetime(df[col], errors="ignore", infer_datetime_format=True)
            else:
                df[col] = df[col].apply(lambda x: "".join([i for i in x if ord(i.lower()) in range(97, 123) or ord(i) in range(48, 58)]))
                if col == "last_name" or col == "dept":
                    df[col] = df[col].apply(lambda x: x.upper())
                elif col == "salary":
                    df[col] = df[col].apply(lambda x: "".join([str((int(i)%10+7)%10) if x.startswith("3X") else i for i in x.replace("3X", "")]))
        df["hire_date"] = df["hire_date"].dt.date

        # Create a column "Name" with LAST_name, First_name, and Middle initial.
        df["name"] = [df.loc[n, "last_name"]+", "+df.loc[n, "first_name"]+", "+df.loc[n, "middle_initial"]+"." if df.loc[n, "middle_initial"] else df.loc[n, "last_name"]+", "+df.loc[n, "first_name"] for n in df.index]
        df.fillna("", inplace=True)
        return df
    
    def emp_list_df(self):
        '''
        Create an alphabetic list of employees by name
        '''
        df = self.data()
        res = df[["name", "gender", "s_g", "salary", "dept", "hire_date", "term_date"]].sort_values(by=["name"], ascending=True)
        return res
    
    def emp_counts_by_dept_page(self):
        '''
        1. Create a list of employees for each department,
        2. And horizontal bar chart with all depts included
        '''

        df = self.emp_counts_dept_df()
        res = self.chart_barh_multi(df, title="Employee Count by Department", groups=df.index)

        return res
    
    def emp_counts_sg_df(self):
        '''
        Create a DataFrame for each Salary Grade with Total Count of Employees by Gender
        '''
        df = self.emp_list_df()
        grades = df.s_g.unique()
        grades.sort()
        emp_counts_sg = pd.DataFrame(index=grades, columns=["Female", "Male"])
        emp_counts_sg.index.name="salary grade"
        for grade in grades:
            female = df[(df["s_g"] == grade)&(df["gender"] == "F")]["salary"].count()
            male = df[(df["s_g"] == grade)&(df["gender"] == "M")]["salary"].count()
            emp_counts_sg.loc[grade] = [female, male]
        return emp_counts_sg
    
    def emp_counts_dept_df(self):
        '''
        Create a DataFrame for each Dept with Total Count of Employees
        '''
        df = self.df

        emp_counts_dept = pd.DataFrame(index=self.dept, columns=["emp_count"])

        for dept in self.dept:
            emp_counts_dept.loc[dept] = [len(df[df['dept'] == dept].index)]
        return emp_counts_dept
        
    def emp_counts_dept_gen_df(self):
        '''
        Create a DataFrame for each Dept with Total Count of Employees Grouped by Gender
        '''
        df = self.emp_list_df()
        departments = df.dept.unique()
        departments.sort()
        emp_counts_dept = pd.DataFrame(index=departments, columns=["Female", "Male"])
        emp_counts_dept.index.name="dept"
        for dept in departments:
            female = len(df[(df["dept"] == dept)&(df["gender"] == "F")].index)
            male = len(df[(df["dept"] == dept)&(df["gender"] == "M")].index)
            emp_counts_dept.loc[dept] = [female, male]
        return emp_counts_dept

    def emp_count_page(self):
        """
        1. Create a DataFrame for each Salary Grade with Total Count of Employees by Gender
        2. Create a DataFrame for each Salary Grade with Total Count of Employees by Dept
        3. Create a Pie Chart with Total Count of Employees by Gender for Each Salary Grade
        4. Create a Pie Chart with Total Count of Employees by Gender for Each Dept
        """
        res = dict()
        # 1
        res['emp_counts_sg'] = self.emp_counts_sg_df()
        # 2
        res['emp_counts_dept_gen'] = self.emp_counts_dept_gen_df()

        # 3
        res['pie_charts_sg']=[]
        for grade in res['emp_counts_sg'].index:
            female, male = res['emp_counts_sg'].loc[grade, "Female"], res['emp_counts_sg'].loc[grade, "Male"]
            pie_url_sg = self.chart_pie(female, male, title=f"Employee Count by Gender for Salary Grade {grade}", explode=[0, 0.1], labels=[f"Female: {female}", f"Male: {male}"], colors=("gold", "royalblue"))
            res['pie_charts_sg'].append(pie_url_sg)

        # 4
        res['pie_charts_dept']=[]
        for dept in res['emp_counts_dept_gen'].index:
            female, male = res['emp_counts_dept_gen'].loc[dept, "Female"], res['emp_counts_dept_gen'].loc[dept, "Male"]
            pie_url_dept = self.chart_pie(female, male, title=f"Employee Count by Gender for Salary Grade {grade}", explode=[0, 0.1], labels=[f"Female: {female}", f"Male: {male}"], colors=("palevioletred", "deepskyblue"))
            res['pie_charts_dept'].append(pie_url_dept)
        
        return res



    def dept_file_df(self):
        '''
        Import and Clean the dept_CAPSTONE file
        '''
        dept_df = pd.read_csv(self.path+"/dept_CAPSTONE.txt", names=["dept_code", "dept_name"], index_col=False)
        dept_df.fillna("", inplace=True)
        dept_df.drop([0], inplace=True)
        for col in dept_df.columns:
            dept_df[col] = dept_df[col].apply(lambda x: "".join([i if ord(i.lower())in range(97, 123) or i == "." or i == " " else "" for i in x]))
            if col == "dept_code":
                dept_df[col] = dept_df[col].apply(lambda x: x.upper())
            else:
                dept_df[col] = dept_df[col].apply(lambda x: x.title())
        return dept_df
    
    def emp_list_with_dept(self):
        '''
        Merges df with all employees and corresponding dept_code & dept_name
        '''
        emp_list = self.emp_list_df()
        dept_file = self.dept_file_df()
        return emp_list.merge(dept_file, left_on=emp_list.dept, right_on=dept_file.dept_code)

    def emp_active_df(self):
        '''
        Create a list of All Active Employees
        '''
        emp_list = self.emp_list_df()
        [emp_list.drop([n], inplace=True) for n in emp_list.index if emp_list.loc[n, "term_date"] != " "]
        return emp_list

    def active_emp(self):
        '''
        1. Filter our the employees who had left the organization
        2. Adds a column 'tenure' and fills it with the duration of service for each active employee.
        3. Group employees by tenure and creates a histgram per dept.
        4. returns a dict containing all the data.
        '''
        # 1
        data = {}
        today = pd.Timestamp.today().date()
        urls = {}
        dept_name = self.dept_dict()
        data['dept_name'] = dept_name
        emp_active = self.emp_active_df()
        emp_active.sort_values(by="hire_date", ascending=True, inplace=True)

        # 2
        for n in emp_active.index:
            emp_active.loc[n, 'tenure'] = round((today - pd.to_datetime(emp_active.loc[n, "hire_date"]).date()).days/365)
        
        # 3
        for dept in self.dept:
            years = emp_active[emp_active.dept == dept].sort_values(by=["tenure"]).tenure.unique()
            emp_active_dept = emp_active[emp_active.dept == dept]
            title = f"{dept}: Active Employee Count by Years On The Job"
            urls[dept] = self.histgram_multi(emp_active_dept, title=title, groups=years, ylim_top=25, column='tenure', target='name', bins=20)

        data['emp_active'] = emp_active
        data['dept'] = self.dept
        data['urls'] = urls
        # 4
        return data


    def dept_dict(self):
        '''
        Create a dictionary of dept code and name pairs
        '''
        res = {}
        depts = self.dept_file_df()
        for dpt in depts.dept_code.unique():
            res[dpt] = depts[depts.dept_code == dpt].dept_name.to_list()[0]
        return res
    
    def emp_status_df(self, data):
        '''
        Adds appropriate value to the column 'status
        '''
        for num in data.index:
            if int(data.loc[num, 's_g']) in range(5,8):
                data.loc[num, 'status'] = 'EXECUTIVE'
            else:
                data.loc[num, 'status'] = 'NON-EXECUTIVE'
        return data

    def emp_id_df(self):
        '''
        Generates employee id (i.e., Last_name[:3]+First_name[:3]+3_digits)
        Create new column 'emp_id' and fills it with the unique id
        '''
        df = self.emp_active_df()
        all_digits = random.sample(["00"+str(n) for n in range(1,10)]+[str(num) if num > 99 else "0"+str(num) for num in range(10,1000)], 617)
        df['emp_id'] = ""
        for n in range(len(all_digits)):
            name = df.iloc[n, 0].replace(" ", "").split(",")
            df.iloc[n, 7] = name[0][:3]+name[1][:3].upper()+all_digits[n]
        df.sort_values(by=["emp_id"], ascending=True, inplace=True)
        return df

    def raise_schedule_df(self):
        '''
        Loads and creates a df from the file containing raise schedule
        '''
        df = pd.read_csv(self.path+"/raises_CAPSTONE.txt")
        df.raise_amount = df.raise_amount.apply(lambda x: x.strip("'"))
        df["ratio"] = round(df.raise_amount.apply(lambda x: float(x.strip("%")))/100, 4)
        df.loc[6, "years"], df.loc[6,"ratio"] = 0, 0
        return df

    def raise_years(self, var_x):
        '''
        Takes an employees tenure and converts it to a numeric value that is consistent with the raise schedule
        '''
        if var_x > 10 or var_x in range(7,10):
            var_x = 10
        elif var_x in range(5,7):
            var_x = 5
        elif var_x == 4:
            var_x = 3
        return var_x

    def calc_raise_df(self):
        '''
        Calculates the raise amount and salary after raise. Simultaneously creates new columns 'raise_amount' and 'new_salsry'.
        Fills them with the calculated values
        '''
        df = self.emp_id_df()
        raise_df = self.raise_schedule_df()
        for n in df.index:
            today = pd.Timestamp.today().date()
            years = self.raise_years(round((today - df.loc[n, "hire_date"]).days/365))
            df.loc[n, "raise_amount"] = round(float(df.loc[n, "salary"])*float(raise_df[raise_df.years == years].ratio),2)
            df.loc[n, "new_salary"] = float(df.loc[n, "salary"])+df.loc[n, "raise_amount"]
        return df

    def salary_range_df(self):
        '''
        Reads and cleans the file containing salary schedule. Then creates a df with starting and ending salary amonut for each salary grade.
        '''
        df = pd.read_csv(self.path+"/salary_grade_CAPSTONE.txt", names=["sg", "start", "end"], index_col=False)
        df.drop([0], inplace=True)
        df.fillna("", inplace=True)
        for col in ["start", "end"]:
            df[col] = df[col].apply(lambda x: "".join([i for i in x if ord(i) in range(48,58)]))
        return df

    def histgram(self, data, **kwargs):
        '''
        Creates histgram, saves it as an image and returns the url
        '''
        ax, fig, col, key, target, bin = kwargs['ax'], kwargs['fig'], kwargs["col"], kwargs["key"], kwargs["target"], kwargs["bin"]
        ax.hist(height=data[data[col] == key][target].count(), x=key, bins=bin)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        image = base64.b64encode(buf.getbuffer()).decode("ascii")
        url = f"img src=data:image/png;base64,{image}"
        return url
    
    def histgram_multi(self, df, **kwargs):
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(1,1,1)
        ax.set_title(label=kwargs['title'], fontdict={'color':"black", 'fontsize':20})
        ax.set_ylim(top=kwargs['ylim_top'])
        for group in kwargs['groups']:
            data = df[df[kwargs['column']] == group][kwargs['target']].count()
            ax.hist(height=data, x=group, bins=kwargs['bins'])
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            image = base64.b64encode(buf.getbuffer()).decode("ascii")
            url = f"img src=data:image/png;base64,{image}"
        return url

    '''
    title = f"{dept}: Active Employee Count by Years On The Job"
            fontdict = {'color':"black", 'fontsize':20}
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(1,1,1)
            ax.set_title(label=title, fontdict=fontdict)
            ax.set_ylim(top=25)
            for year in years:
                res = func.histgram(emp_active_dept, ax=ax, fig=fig, col="tenure", key=year, target="name", title=title, fontdict=fontdict, bin=20)
                urls[dept] = res
    '''

    def unique_values(self, data, target):
        '''
        Generates a list of unique values in a column
        '''
        return data.sort_values(by=[target])[target].unique()
    
    def chart_pie1(self, data_dict):
        '''
        Takes a dict of parameters to generate a pie chart.
        Creates an image and return url
        '''
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(1,1,1)
        ax.set_title(label=data_dict['title'], fontdict={'color':"black", 'fontsize':20})
        ax.pie(data_dict['data'].values(), explode=data_dict['explode'], autopct="%1.2f%%", labels=data_dict['labels'], textprops={'color':"black", 'size':13}, colors=(i for i in data_dict['colors']))
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        image = base64.b64encode(buf.getbuffer()).decode("ascii")
        url = f"img src=data:image/png;base64,{image}"
        return url
    
    def scatter_plot(self, var_x, var_y, title, colors):
        '''
        Generates a scatter plot, creates an image, and returns the url
        '''
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(1,1,1)
        ax.set_title(label=title, fontdict={'color':"black", 'fontsize':20})
        ax.scatter(var_x, var_y, c=colors)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        image = base64.b64encode(buf.getbuffer()).decode("ascii")
        url = f"img src=data:image/png;base64,{image}"
        return url
    
    def chart_barh(self, data):
        '''
        Takes a dict of parameters to generate a horizontal bar chart.
        Creates an image and return url
        '''
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(1,1,1)
        ax.set_ylim(top=data['ylim']) if data['ylim'] else None
        ax.set_title(label=data['title'], fontdict={'color':"black", 'fontsize':20})
        ax.barh(width=data['width'], y=data['ylabel'], height=data['height'], color=data['color'])
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        image = base64.b64encode(buf.getbuffer()).decode("ascii")
        url = f"img src=data:image/png;base64,{image}"
        return url
    
    def chart_barh_multi(self, df, **kwargs):
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(1,1,1)
        ax.set_title(label=kwargs['title'], fontdict={'color':"black", 'fontsize':20})
        for group in kwargs['groups']:
            data = df.loc[group]
            ax.barh(width=data, y=group)
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            image = base64.b64encode(buf.getbuffer()).decode("ascii")
            url = f"img src=data:image/png;base64,{image}"
        return url

    def chart_bar(self, data):
        '''
        Takes a dict of parameters to generate a bar chart.
        Creates an image and return url
        '''
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(1,1,1)
        ax.set_title(label=data['title'], fontdict={'color':"black", 'fontsize':20})
        ax.bar(x=data['xlabel'], height=data['data'].values(), color=['orangered', 'lightcoral', 'gold', 'lightseagreen',  'cornflowerblue', 'mediumpurple', 'slategray'])
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        image = base64.b64encode(buf.getbuffer()).decode("ascii")
        url = f"img src=data:image/png;base64,{image}"
        return url
    
    def chart_distribution(self, data, title, chart_color):
        '''
        Generates a distribution chart.
        Creates an image and return url
        '''
        data = pd.Series(data)
        data = data.apply(lambda x: float(x))
        stats = Stats.basic_stats(data)
        fig_plot = Figure(figsize=(13, 7))
        ax_plot = fig_plot.add_subplot(1,1,1)
        ax_plot.set_title(label=title, fontdict={'color':"black", 'fontsize':20})
        cnt, bin_n, patches = ax_plot.hist(data, 80, color=chart_color)
        ax_plot.plot(bin_n[:-1], cnt, '--', color='dimgray')
        stats = f"Mean: {stats['mean']}\nMedian: {stats['median']}\nMode(s): {stats['modes']}\nSD: {stats['sd']}"
        ax_plot.text(0.85, 0.73, stats, ha='right', va='top', transform=fig_plot.transFigure,fontdict={"fontsize": 13})
        buf = io.BytesIO()
        fig_plot.savefig(buf, format="png")
        image = base64.b64encode(buf.getbuffer()).decode("ascii")
        url = f"img src=data:image/png;base64,{image}"
        return url
    
    def chart_pie(self, *args, **kwargs):
        '''
        Takes data as args and other parameters as kwargs and creates a pie chart and url for the image
        '''
        fig_pie = Figure(figsize=(8, 5))
        ax_pie = fig_pie.add_subplot(1,1,1)
        ax_pie.set_title(label=kwargs['title'], fontdict={'color':"black", 'fontsize':20})
        ax_pie.pie(args, explode=kwargs['explode'], autopct="%1.2f%%", labels=kwargs['labels'], textprops={'color':"black", 'size':17}, colors=(i for i in kwargs['colors']))
        buf = io.BytesIO()
        fig_pie.savefig(buf, format="png")
        image = base64.b64encode(buf.getbuffer()).decode("ascii")
        url = f"img src=data:image/png;base64,{image}"
        return url

    def scatter_plot_dict_for_emp_salary(self):
        df = self.emp_status_df(self.emp_active)
        all_salaries = self.chart_distribution(df.salary, "Salary Distribution: ALL", 'limegreen')
        salaries_men = self.chart_distribution(df[df.gender == "M"].salary, "Salary Distribution: MEN", 'deepskyblue')
        salaries_women = self.chart_distribution(df[df.gender == "F"].salary, "Salary Distribution: WOMEN", 'hotpink')
        stats = dict()
        stats["scatter_plots"]= dict()
        stats["scatter_plots"]["mean_by_sg"] = dict()
        mean_salary_sg = list()
        mean_salary_sg_gender = dict()

        for gen in self.genders:
            stats[gen] = Stats.basic_stats(df[df.gender == gen].salary)
        
        for grade in self.sg:
            stats[grade] = dict()
            stats[grade]["ALL"] = Stats.basic_stats(df[df.s_g == grade].salary)
            mean_salary_sg.append(stats[grade]["ALL"]["mean"])
        
        stats["scatter_plots"]["mean_by_sg"]["ALL"] = self.scatter_plot(self.sg, mean_salary_sg, "Mean Salary by Salary Grade", ["red"])

        for gen in self.genders:
            mean_salary_sg_gender[gen] = []
        for grade in self.sg: 
            '''Computes basic stats values for each salary grade grouped by gender'''
            stats[grade][gen] = Stats.basic_stats(df[(df.s_g == grade)&(df.gender == gen)].salary)
            mean_salary_sg_gender[gen].append(stats[grade][gen]["mean"])
        
        for status in df.status.unique():
            stats[status] = dict()
            '''Computes basic stats values for all employees' salaries grouped by employment status'''
            stats[status]["ALL"] = Stats.basic_stats(df[df.status == status].salary)

            for gen in df.gender.unique():
                '''Computes basic stats values for all employees' salaries grouped by employment status and gender'''
                stats[status][gen] = Stats.basic_stats(df[(df.status == status)&(df.gender == gen)].salary)

            '''creates a pie chart with the numbers of employees for each employment status grouped by gender. Stores the chart's url to the dict "stats"'''
            stats[status]["chart_pie"] = self.chart_pie(stats[status]["M"]["sample_size"], stats[status]["F"]["sample_size"], title=f"'{status}' Employee Count by Gender", explode=[0, 0.1], labels=["Male", "Female"], colors=["deepskyblue", "hotpink"])

            '''creates a horizontal bar chart with the mean salaries for each employment status grouped by gender. Stores the chart's url to the dict "stats"'''
            stats[status]["chart_barh_gender"] = self.chart_barh({'width':[stats[status]["M"]["mean"], stats[status]["F"]["mean"]], 'ylabel':["Male", "Female"], 'height':0.8, 'title':status, 'color':["deepskyblue", "hotpink"], 'ylim':None})
        
        '''Sorts the df containing all active employees by salary grade then name in alphabetica order'''
        all_emp = df.sort_values(by=["s_g", "name"])
        res = dict()

        res['all_salaries'] = all_salaries
        res['salaries_men'] = salaries_men
        res['salaries_women'] = salaries_women
        res['stats'] = stats
        res['sg'] = self.sg
        res['emp_status'] = df.status.unique()
        res['genders'] = self.genders
        res['all_emp'] = all_emp
        
        return res

    def salary_raise_page(self):
        res = dict()
        df = self.df_emp_id
        res['df'] = df
        raise_df = self.df_raise

        for n in df.index:
            today = pd.Timestamp.today().date()
            years = self.raise_years(round((today - df.loc[n, "hire_date"]).days/365))
            df.loc[n, "raise_amount"] = round(float(df.loc[n, "salary"])*float(raise_df[raise_df.years == years].ratio),2)
            df.loc[n, "new_salary"] = float(df.loc[n, "salary"])+df.loc[n, "raise_amount"]
        
        info = {'new_salary': {}, 'raise_total': {}, 'raise_by_gender': {}, 'raise_by_gender_dept': {}}
        salary_by_dept, raise_by_dept, raise_by_gender, raise_by_gender_dept = dict(), dict(), dict(), dict()
        res['url_raise_by_gender_dept'] = list()
        for dept in self.dept:
            raise_by_gender_dept[dept] = dict()
            salary_by_dept[dept] = df[df.dept == dept].new_salary.sum()
            raise_by_dept[dept] = df[df.dept == dept]['raise_amount'].sum()
            for gender in df.gender.unique():
                raise_by_gender[gender] = df[df.gender == gender].raise_amount.sum()
                raise_by_gender_dept[dept][gender] = \
                    (df[(df.gender == gender)&(df.dept == dept)].raise_amount.sum())
            res['url_raise_by_gender_dept'].append(\
                self.chart_pie(raise_by_gender_dept[dept]['F'], \
                    raise_by_gender_dept[dept]['M'], \
                        title=f"{dept}: Total Raise Allocation by Gender", \
                            colors=["deepskyblue", 'hotpink'], \
                                explode=[0, 0.1], labels=['Female', 'Male']))
        
        info['new_salary']['data'] = salary_by_dept
        info['raise_total']['data'] = raise_by_dept      
        info['raise_by_gender']['data'] = raise_by_gender
            
        info['new_salary']['title'] = "Total New Salary per Dept"
        info['new_salary']['xlabel'] = list(info['new_salary']['data'].keys())
        res['url_new_salary'] = self.chart_bar(info['new_salary'])
            
        info['raise_total']['title'] = "Sum of Raise Amount per Dept"
        info['raise_total']['colors'] = ['orangered', 'lightcoral', 'gold', 'lightseagreen',  'cornflowerblue', 'mediumpurple', 'slategray']
        info['raise_total']['explode'] = None
        info['raise_total']['labels'] = list(info['raise_total']['data'].keys())
        res['url_raise_all'] = self.chart_pie1(info['raise_total'])

        info['raise_by_gender']['title'] = "Total Raise Amount by Gender"
        info['raise_by_gender']['colors'] = ["deepskyblue", 'hotpink']
        info['raise_by_gender']['explode'] = [0, 0.1]
        info['raise_by_gender']['labels'] = list(info['raise_by_gender']['data'].keys())
        res['url_raise_gender'] = self.chart_pie1(info['raise_by_gender'])

        return res

    def salary_promo_page(self):
        '''
        Checks each employee's new salary and salary grade to see if one is ready for promotion
        '''
        df = self.calc_raise_df()
        salary_range = self.salary_range_df()
        for n in df.index:
            salary, sg = df.loc[n, "new_salary"], df.loc[n, "s_g"]
            if sg != "7":
                if salary > float(salary_range[salary_range.sg == sg]["end"]):
                    df.loc[n, "promotion"] = "PROMOTION DUE"
        df.fillna("", inplace=True)
        # ready = df[df.promotion == "PROMOTION DUE"].emp_id.count()

        res = dict()
        promo_by_sg_gender = dict()
        promo_by_sg_gender["ALL"] = dict()
        promo_by_sg_gender["ALL"]['data'] = dict()
        url_promo_by_sg_gender = dict()

        grades = df.s_g.unique()[:-1] # All salary grades but 7
        for grade in grades:
            promo_by_sg_gender[grade] = dict()
            promo_by_sg_gender[grade]['data'] = dict()
            for gender in df.gender.unique():
                promo_by_sg_gender["ALL"]['data'][gender] = df[(df.promotion == "PROMOTION DUE")&(df.gender == gender)].emp_id.count()
                promo_by_sg_gender[grade]['data'][gender] = df[(df.s_g == grade)&(df.promotion == "PROMOTION DUE")&(df.gender == gender)].emp_id.count()
            promo_by_sg_gender[grade]['title'] = f"Salary Grade {grade}: Employees Due to Promotion by Gender"
            promo_by_sg_gender[grade]['colors'] = ["deepskyblue", 'hotpink']
            promo_by_sg_gender[grade]['explode'] = [0, 0.1]
            promo_by_sg_gender[grade]['labels'] = ['Female', 'Male']
            url_promo_by_sg_gender[grade] = self.chart_pie1(promo_by_sg_gender[grade])
        
        promo_by_sg_gender["ALL"]['title'] = "Employees Due to Promotion by Gender"
        promo_by_sg_gender["ALL"]['colors'] = ["deepskyblue", 'hotpink']
        promo_by_sg_gender["ALL"]['explode'] = [0, 0.1]
        promo_by_sg_gender["ALL"]['labels'] = ['Female', 'Male']
        by_gender_all = self.chart_pie1(promo_by_sg_gender["ALL"])
        res['df'] = df
        res['by_gender_all'] = by_gender_all
        res['url_promo_by_sg_gender'] = url_promo_by_sg_gender
        res['sg'] = grades

        return res

class HealthCarePortal:
    LIMIT = None
    GENDERS = ["F", "M"]

    def __init__(self, path):
        self.path = path+"/data"
        self.description = ProjectDescription.project_description(self.path+'/health_care_portal_description.txt')

        if not self.__is_empty(MedCode):
            self.med_code_table()
        if not self.__is_empty(Employee):
            self.emp_table()
        if not self.__is_empty(Transactions):
            self.transactions_table()


    def med_code_table(self):
        df = pd.read_excel(self.path+"/2021_medical_codes.xlsx", index_col=False, usecols=["CODE","DESCRIPTION","CATEGORY"])
        for col in ["CODE", "DESCRIPTION", "CATEGORY"]:
            if col == "CODE":
                df[col] = df[col].apply(lambda x: str(x).strip(" "))
            else:
                df[col] = df[col].apply(lambda x: "".join([i for i in x.lower() if ord(i) in range(97, 123) or ord(i) in range(47, 58) or i == " "]).title())
        for num in df.index:
            session.add(MedCode(code = df.loc[num, "CODE"], description = df.loc[num, "DESCRIPTION"], category = df.loc[num, "CATEGORY"]))
            session.commit()

    def emp_table(self):
        df = pd.read_csv(self.path+"/patient_accounts.txt", index_col=False, names=["emp_id", "title", "gender", "last_name", "first_name", "salary", "city", "state"])
        df.fillna("", inplace=True)
        for col in df.columns:
            df[col] = df[col].apply(lambda x: "".join([i for i in x if ord(i.lower()) in range(97, 123) or ord(i) in range(48, 58) or i == "%" or i == " " or i == "-"]))
            if col == "salary":
                df[col] = df[col].astype(int)
            elif col == "city":
                df[col] = df[col].apply(lambda x: x.replace("%", " ").title())
            elif col == "first_name":
                df[col] = df[col].apply(lambda x: x.strip(" ").title())

        df.to_sql("patient_accounts", con=engine, if_exists="append", index="id")

    def transactions_table(self):
        df = pd.read_csv(self.path+"/patient_transactions.csv", names=["emp_id", "trans_id", "procedure_date", "medical_code", "procedure_price"], converters={"procedure_price": float}, parse_dates=["procedure_date"])

        for col in ["emp_id", "trans_id", "medical_code"]:
            df[col] = df[col].apply(lambda x: "".join([i for i in x if ord(i) in range(48, 58) or ord(i.lower()) in range(97, 123) or i == "-"]))
        df.to_sql("patient_transactions", con=engine, if_exists="append", index="id")
    
    def med_code_search(self, input):
        res = dict()
        data = session.query(MedCode).filter(MedCode.code == input)
        if data.first():
            res['code'], res['description'], res['category'] = data[0].code, data[0].description, data[0].category
        else:
            return render_template("med_codes.html", message="Invalid Medical Code")
        
        return render_template("med_codes_result.html", code=res['code'], description=res['description'], category=res['category'])

    
    def med_descript_search(self, input):
        if len(input) > 1:
            data = session.query(MedCode).filter(MedCode.description.like(f"%{input}%"))
            if data.first():
                data = data.all()
                result = f"{len(data)} records found matching '{input}'"
                return render_template("med_descript_result.html", data=data, result=result)
            else:
                return render_template("med_descript.html", message="Invalid Key Word")
        else:
            return render_template("med_descript.html", message="Invalid Key Word")

    def emp_search_by_id(self, input):
        if len(input) < 2:
            res = "Invalid Employee ID"
            return render_template("emp_search.html", message=res)
        else:
            data = session.query(Employee).filter(Employee.emp_id == input)
            if data.first():
                data = data.all()
                res = f"{len(data)} record(s) found matching '{input}'"
                return render_template("emp_search_result.html", data=data, result=res)
            else:
                res = f"No Employee Found For: {input}"
                return render_template("emp_search.html", message=res)
            
    def emp_search_by_last(self, input):
        if len(input) < 2:
            res = "Invalid Last Name"
            return render_template("emp_search_last.html", message=res)
        
        else:
            data = session.query(Employee).filter(Employee.last_name.like(f"%{input}%"))
            if data.first():
                data = data.limit(self.LIMIT).all() if self.LIMIT else data.all()
                res = f"{len(data)} record(s) found matching '{input}'"
                return render_template("emp_search_last_result.html", data=data, result=res)
            else:
                res = "Invalid Last Name"
                return render_template("emp_search_last.html", message=res)
    
    def emp_search_salary_range(self, input_min, input_max):
        if len(input_min) < 2 or len(input_max) < 2:
            res = "Invlaid Range"
            return render_template("emp_search_salary_range.html", message=res)
        else:
            start = int(input_min+"000") if len(input_min) in range(2,4) else int(input_min)
            end = int(input_max+"000") if len(input_max) in range(2,4) else int(input_max)
            data = session.query(Employee).filter(and_(Employee.salary >= start, Employee.salary < end))
            if data.first():
                data = data.order_by(desc(Employee.salary)).all()
                res = f"{len(data)} employee(s) found for salaries between ${start} and ${end}"
                return render_template("emp_search_salary_range_result.html", data=data, result=res)
            else:
                res = "Invalid Range"
                return render_template("emp_search_salary_range.html", message=res)

    def emp_portal(self, input_id):
        '''
        param: a complete employee id which == 10 in length
        '''
        if len(input_id) != 10:
            res = "Invalid Employee ID"
            return render_template("employee_portal.html", message=res)
        else:
            data_emp = session.query(Employee).filter(Employee.emp_id == input_id)
            data_transaction = data2 = session.query(Transactions).filter(Transactions.emp_id == input_id)
            if data_emp.first():
                count = data_transaction.count()
                total = session.query(func.sum(Transactions.procedure_price)).filter(Transactions.emp_id == input_id)[0][0]
                data_emp = data_emp.all()[0]
                data_transaction = data_transaction.order_by(asc(Transactions.procedure_date)).all()
                res = f"{len(data_transaction)} transaction record(s) found for employee {input_id}"

                return render_template("employee_portal_result.html", data_emp=data_emp, data_transaction=data_transaction, result=res, count=count, total=total)
    
            else:
                return render_template("employee_portal.html", message="Invalid Employee ID")
    
    def hr_portal(self, input_info):
        res = dict()
        data_emp_and_transaction = session.query(Employee, Transactions).filter(and_(Transactions.medical_code == input_info, Employee.emp_id == Transactions.emp_id)).order_by(Transactions.procedure_date)
        data_med = session.query(MedCode).filter(MedCode.code == input_info).limit(1)[0]
        if data_emp_and_transaction.first():
            for gender in self.GENDERS:
                emp_gender = "Female" if gender == "F" else "Male"
                mean = session.query(func.avg(Transactions.procedure_price)).filter(and_(Transactions.medical_code == input_info, Employee.emp_id == Transactions.emp_id, Employee.gender == gender[0])).all()[0][0]
                res[gender] = f"Average Cost per {emp_gender} Patient: ${round(mean, 2)}"
            data_emp_and_transaction = data_emp_and_transaction.all()
            med_code = f"{data_med.code}: {data_med.description}"
        
            return render_template("hr_portal_med_code_result.html", data=data_emp_and_transaction, result=med_code, male_avg=res["M"], female_avg=res["F"])
            
        else:
            return render_template("hr_portal_med_codel.html", message="Invalid Medical Code")
    
    def hr_summary(self):
        res = dict()
        years = session.query(extract("year", Transactions.procedure_date)).distinct().order_by(extract("year", Transactions.procedure_date)).all()
        for num in range(len(years)):    
            data = session.query(func.count(Employee.emp_id.distinct()), func.count(Transactions.trans_id), func.sum(Transactions.procedure_price), func.avg(Transactions.procedure_price)).filter(and_(extract('year', Transactions.procedure_date) == years[num][0]), Employee.emp_id == Transactions.emp_id).all()[0]
            all_price = session.query(Transactions.procedure_price).filter(and_(extract('year', Transactions.procedure_date) == years[num][0])).order_by(asc(Transactions.procedure_price)).all()
            prices = [all_price[num][0] for num in range(len(all_price))]
            median = Stats.find_median(prices)
            modes = Stats.find_modes(prices)
            sd = round(np.array(prices).std(), 2)
            res[years[num][0]] = [data[0], data[1], data[2], round(data[3], 2), median, modes, sd]
        
        return res

    def hr_benefit_cost(self):
        res = dict()
        titles = session.query(Employee.title).distinct().order_by(Employee.title).all()
        years = session.query(extract("year", Transactions.procedure_date)).distinct().order_by(extract("year", Transactions.procedure_date)).all()
        for title_num in range(len(titles)):
            lst = []
            for year_num in range(len(years)):
                data_by_year = dict()
                data = session.query(func.count(Employee.emp_id.distinct()), func.count(Transactions.trans_id), func.sum(Transactions.procedure_price), func.avg(Transactions.procedure_price)).filter(and_(extract('year', Transactions.procedure_date) == years[year_num][0]), Employee.emp_id == Transactions.emp_id, Employee.title == titles[title_num][0]).all()[0]
                all_price = session.query(Transactions.procedure_price).filter(and_(extract('year', Transactions.procedure_date) == years[year_num][0])).order_by(asc(Transactions.procedure_price)).all()
                prices = [all_price[n][0] for n in range(len(all_price))]
                median = Stats.find_median(prices)
                modes = Stats.find_modes(prices)
                sd = round(np.array(prices).std(), 2)
                data_by_year[years[year_num][0]] = [data[0], data[1], data[2], round(data[3], 2), median, modes, sd]
                lst.append(data_by_year)
            res[titles[title_num][0]] = lst
        return res

    def medical_cost_analysis(self):
        res = dict()
        proc_price = dict()
        all_codes = list()
        for gender in self.GENDERS:
            codes = session.query(Transactions.medical_code).filter(and_(Transactions.emp_id == Employee.emp_id, Employee.gender == gender)).distinct().order_by(Transactions.medical_code).all()
            all_codes.append(codes)

        codes = sorted(set(all_codes[0]) & set(all_codes[1]))
        codes = [i[0] for i in codes]
        res['all_codes'] = codes
        res['genders'] = self.GENDERS
        for code in codes:
            res[code] = dict()
            description = session.query(MedCode.description).filter(MedCode.code == code).all()[0][0]
            for gender in self.GENDERS:
                emp_gender = 'female' if gender == "F" else 'male'
                proc_price[emp_gender] = session.query(Transactions.procedure_price).filter(and_(Transactions.emp_id == Employee.emp_id, Employee.gender == gender, Transactions.medical_code == code)).all()

            pricesM = [round(proc_price['male'][num][0], 4) for num in range(len(proc_price['male']))]
            pricesF = [round(proc_price['female'][num][0], 4) for num in range(len(proc_price['female']))]
            meanM, meanF = round(Stats.calc_mean(pricesM),2), round(Stats.calc_mean(pricesF),2)
            medianM, medianF = round(Stats.find_median(pricesM),4), round(Stats.find_median(pricesF),4)
            modesM, modesF = Stats.find_modes(pricesM), Stats.find_modes(pricesF)
            sdM, sdF = round(Stats.calc_SD(pricesM),4), round(Stats.calc_SD(pricesF),4)
            diffM, diffF = round((meanM - meanF),4), round((meanF - meanM),4)
            data = {"desc": description, "M": [meanM, medianM, modesM, sdM, diffM], "F": [meanF, medianF, modesF, sdF, diffF]}
            res[code]['data'] = data
            res[code]['diff'] = diffF
        
        return res
    
    def med_cost_analysis_charts(self):
        result = dict()
        total_by_title = session.query(Employee.title, func.sum(Transactions.procedure_price)).filter(Transactions.emp_id == Employee.emp_id).group_by(Employee.title).all()
        fig_title = Figure(figsize=(11,6))
        ax_title = fig_title.add_subplot(1,1,1)
        ax_title.set_title(label="Total Cost of Procedures By Title", fontdict={'color':"black", 'fontsize':20})
        for num in range(len(total_by_title)):
            res = total_by_title[num][1]
            ax_title.barh(width=res, y=total_by_title[num][0])
            buf = io.BytesIO()
            fig_title.savefig(buf, format="png")
            data_title = base64.b64encode(buf.getbuffer()).decode("ascii")
            res_title = f"data:image/png;base64,{data_title}"
        result['barh_title'] = res_title
        

        total_by_year = session.query(extract("year", Transactions.procedure_date), func.sum(Transactions.procedure_price)).filter(Transactions.emp_id == Employee.emp_id).group_by(extract("year", Transactions.procedure_date)).all()
        fig_year = Figure(figsize=(11,6))
        ax_year = fig_year.add_subplot(1,1,1)
        ax_year.set_title(label="Total Cost of Procedures By Year", fontdict={'color':"black", 'fontsize':20})
        for num in range(len(total_by_year)):
            res = total_by_year[num][1]
            ax_year.bar(total_by_year[num][0], res)
            buf = io.BytesIO()
            fig_year.savefig(buf, format="png")
            data_year = base64.b64encode(buf.getbuffer()).decode("ascii")
            res_year = f"data:image/png;base64,{data_year}"
        result['bar_year'] = res_year

        total_by_year_gender = session.query(extract("year", Transactions.procedure_date), Employee.gender, func.sum(Transactions.procedure_price)).filter(Transactions.emp_id == Employee.emp_id).group_by(extract("year", Transactions.procedure_date)).group_by(Employee.gender).all()
        charts = []
        for num in range(0,len(total_by_year_gender), 2):
            res_female = total_by_year_gender[num][2]
            res_male = total_by_year_gender[num+1][2]
            fig_pie = Figure()
            ax_year_gender = fig_pie.add_subplot(1,1,1, title=f"{total_by_year_gender[num][0]} Total: ${res_male+ res_female}")
            ax_year_gender.pie([res_male, res_female], explode=[0.1, 0], autopct="%1.2f%%", labels=["Male", "Female"], textprops={'color':"black", 'size':15})
            buf = io.BytesIO()
            fig_pie.savefig(buf, format="png")
            data = base64.b64encode(buf.getbuffer()).decode("ascii")
            url = f"data:image/png;base64,{data}"
            charts.append(url)
        result['pie_charts'] = charts

        return result

    def health_care_portal_salary_analysis(self):
        res = dict()
        res['charts'] = dict()
        res['all_stats'] = dict()
        all_salaries = [i[0] for i in session.query(Employee.salary).order_by(Employee.salary).all()]

        res_salary = Charts.chart_hist(all_salaries, 100, "Salary Distribution")
        res['charts']['salary_distribution'] = res_salary
        res_cumulative = Charts.chart_cumulative(all_salaries, 100, "Salary Distribuion Cumulative")
        res['charts']['distribution_cumulative'] = res_cumulative

        mean = round(sum(all_salaries)/len(all_salaries),4)
        median = round(Stats.find_median(all_salaries),4)
        mode = Stats.find_modes(all_salaries)
        sd = round(Stats.calc_SD(all_salaries),4)
        res['all_stats']["all_salary"] = {"title":"All", "gender":"All", "mean": mean, "median":median, "modes": mode, "SD": sd, "Diff": 0}

        titles = [i[0] for i in session.query(Employee.title).distinct()]
        
        for gender in self.GENDERS:
            salaries = [i[0] for i in session.query(Employee.salary).filter(Employee.gender == gender).all()]
            mean_gend = round(sum(salaries)/len(salaries), 4)
            median_gend = round(Stats.find_median(salaries),4)
            mode_gend = Stats.find_modes(salaries)
            sd_gend = round(Stats.calc_SD(salaries),4)
            stats_gend = {"title":"All", "gender":gender, "mean": mean_gend, "median":median_gend, "modes":mode_gend, "SD": sd_gend, "Diff":0}
            res['all_stats'][gender] = stats_gend

        for title in titles:   
            for gender in self.GENDERS:
                salaries_gend_title = salaries = [i[0] for i in session.query(Employee.salary).filter(and_(Employee.gender == gender, Employee.title == title)).all()]
                mean_gend_title = round(sum(salaries_gend_title)/len(salaries_gend_title), 4)
                median_gend_title = round(Stats.find_median(salaries_gend_title),4)
                mode_gend_title = Stats.find_modes(salaries_gend_title)
                sd_gend_title = round(Stats.calc_SD(salaries_gend_title),4)
                key = title+"_"+gender
                res['all_stats'][key] = {"title":title, "gender":gender, "mean": mean_gend_title, "median":median_gend_title, "modes":mode_gend_title, "SD": sd_gend_title, "Diff":0}
        
        all_keys = list(res['all_stats'].keys())
        for num in range(1, len(all_keys), 2):
            res['all_stats'][all_keys[num]]["diff"] = round(res['all_stats'][all_keys[num]]["mean"] - res['all_stats'][all_keys[num+1]]["mean"],4)
        for num in range(2, len(all_keys), 2):
            res['all_stats'][all_keys[num]]["diff"] = round(res['all_stats'][all_keys[num]]["mean"] - res['all_stats'][all_keys[num-1]]["mean"],4)
        
        return res

    def salary_analysis_with_pd(self):
        res = dict()
        res['all_salary_pd'] = dict()
        res['charts'] = dict()

        emp_info = self.__pd_data(self.path+"/patient_accounts.txt", False, ["emp_id", "title", "gender", "last_name", "first_name", "salary", "city", "state"])
        for col in emp_info.columns:
            emp_info[col] = emp_info[col].apply(lambda x: "".join([i for i in x if ord(i.lower()) in range(97, 123) or ord(i) in range(48, 58) or i == "%" or i == " " or i == "-"]))
            if col == "salary":
                emp_info[col] = emp_info[col].astype(int)
            elif col == "city":
                emp_info[col] = emp_info[col].apply(lambda x: x.replace("%", " ").title())
            elif col == "first_name":
                emp_info[col] = emp_info[col].apply(lambda x: x.strip(" ").title())
        mean = emp_info.salary.mean()
        median = emp_info.salary.median()
        modes = Stats.find_modes(emp_info.salary.tolist())
        sd = emp_info.salary.std()
        res['all_salary_pd']["all"] = {"title": "All", "gender": "All", "mean": mean, "median": median, "modes": modes, "SD": sd}
        
        for gender in emp_info.gender.unique():
            data = emp_info[emp_info.gender == gender].salary
            mean, median, modes, sd = data.mean().round(4), data.median().round(4), Stats.find_modes(data.tolist()), data.std().round(4)
            res['all_salary_pd'][gender] = {"title": "All", "gender": gender, "mean": mean, "median": median, "modes": modes, "SD": sd}
        
        for title in emp_info.title.unique():
            for gender in emp_info.gender.unique():
                data = emp_info[(emp_info.title == title)&(emp_info.gender == gender)].salary
                mean, median, modes, sd = data.mean().round(4), data.median().round(4), Stats.find_modes(data.tolist()), data.std().round(4)
                res['all_salary_pd'][title+"_"+gender] = {"title": title, "gender": gender, "mean": mean, "median": median, "modes": modes, "SD": sd}
        all_keys = list(res['all_salary_pd'].keys())
        for num in range(1, len(all_keys), 2):
            res['all_salary_pd'][all_keys[num]]["Diff"] = round(res['all_salary_pd'][all_keys[num]]["mean"] - res['all_salary_pd'][all_keys[num+1]]["mean"],4)
        for num in range(2, len(all_keys), 2):
            res['all_salary_pd'][all_keys[num]]["Diff"] = round(res['all_salary_pd'][all_keys[num]]["mean"] - res['all_salary_pd'][all_keys[num-1]]["mean"],4)

        res_salary = Charts.chart_hist(emp_info.salary, 100, "Salary Distribution")
        res['charts']['salary_distribution'] = res_salary

        res_cumulative = Charts.chart_cumulative(emp_info.salary, 100, "Salary Distribuion Cumulative")
        res['charts']['distribution_cumulative'] = res_cumulative

        return res
        
    
    def med_proc_analysis_sql(self):
        res = dict()
        conn = engine.connect()
        statement_by_count = text("SELECT mc.code, mc.description, pt.cnt as 'Total_Count' FROM med_codes as mc JOIN (SELECT medical_code, Count(medical_code) as 'cnt' FROM patient_transactions GROUP by medical_code ORDER by cnt DESC LIMIT(10)) as pt ON mc.code = pt.medical_code GROUP by mc.code ORDER by Total_Count DESC;")
        top10count = conn.execute(statement_by_count).all()
        res['top10count'] = top10count
        
        statement_by_cost = text("""SELECT mc.code, mc.description, pt.Total_Cost as "Total_Procedure_Cost" FROM med_codes as mc JOIN (SELECT medical_code, Sum(procedure_price) as 'Total_Cost' FROM patient_transactions GROUP by medical_code ORDER by Total_Cost DESC LIMIT(10)) as pt ON mc.code = pt.medical_code GROUP by mc.code ORDER by Total_Procedure_Cost DESC;""")
        top10cost = conn.execute(statement_by_cost).all()
        res['top10cost'] = top10cost
        
        statement_by_emp_cost = text("SELECT pa.emp_id, pa.last_name, pa.first_name, pa.gender, pt.total_cost as 'Total_Medical_Cost' FROM patient_accounts as pa JOIN(SELECT emp_id, SUM(procedure_price) as total_cost FROM patient_transactions GROUP by emp_id ORDER by total_cost DESC LIMIT(10)) as pt ON pa.emp_id = pt.emp_id GROUP by pa.emp_id ORDER by Total_Medical_Cost DESC")
        top10emp = conn.execute(statement_by_emp_cost).all()
        res['top10emp'] = top10emp
        
        conn.close()

        return res


    def __pd_data(self, x, y, z):
        df = pd.read_csv(x, index_col=y, names=z)
        df = df.fillna("")
        return df

    def __is_empty(self, table):
        return session.query(table).first()
    
    def __is_not_empty(self, table):
        if session.query(table).first():
            return True
        return False
    

    def pd_data(self, file, indx, columns):
        df = pd.read_csv(file, index_col=indx, names=columns)
        df = df.fillna("")
        return df

    def chart_hist(self, data, bin, title):
        fig = Figure(figsize=(11, 6))
        ax = fig.add_subplot(1,1,1)
        ax.set_title(label=title, fontdict={'color':'black', 'fontsize':20})
        ax.hist(data, bins=bin)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        url = base64.b64encode(buf.getbuffer()).decode("ascii")
        res = f"img src=data:image/png;base64,{url}"
        return res

    def chart_cumulative(self, data, bin, title):
        fig = Figure(figsize=(11, 6))
        ax = fig.add_subplot(1,1,1)
        ax.set_title(label=title, fontdict={'color':'black', 'fontsize':20})
        ax.hist(data, bins=bin, cumulative=True, histtype="step")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        url = base64.b64encode(buf.getbuffer()).decode("ascii")
        res = f"img src=data:image/png;base64,{url}"
        return res

    def project_description(self, file):
        texts = []
        with open(file, 'r') as file:
            line = file.readline()
            cnt = 0
            while line:
                if len(line) > 5:
                    texts.append(line)
                line = file.readline()
                cnt += 1
        return texts
    

class ProjectDescription:

    def project_description(file):
        texts = []
        with open(file, 'r') as file:
            line = file.readline()
            cnt = 0
            while line:
                if len(line) > 5:
                    texts.append(line)
                line = file.readline()
                cnt += 1
        return texts


class JobDescriptionCleaner:

    '''
    1. set the working directory path
    2. specify the file that contains words to remove from job description
    3. set the symbols/characters to be removed from texts
    4. specify the file that contains unncessary words to be removed from the resume (e.g., 'Location', 'Company')
    \ to improve the accuracy of match score
    5. set the folder where job descriptions are stored
    '''
    
    CONJUNC_ADJECT = "/conjunctions_and_unnecessary_words.txt"
    REPLACE = [':', '.', ',', '(', ')']
    IRRELEVANT = '/remove_from_resume.txt'

    def __init__(self, description_file):
        '''
        1. reads the file that contains words to be removed and turns them into a list
        2. asks for the input file path (i.e., the job description)
        3. clean the job description and turns it into a list of words
        4. count how many times each word appeared in the job description
        '''
        self.ignore = self.conjunc_adject_list()
        self.infile = description_file
        self.cleaned_description = self.remove_conjunc_adject()
        self.word_counted_description = self.count_words(self.cleaned_description)

    def conjunc_adject_list(self):
        '''
        reads the file that contains words to be removed and turns them into a list
        returns a list of those words
        '''
        file_path = PATH+self.CONJUNC_ADJECT
        with open(file_path, 'r') as file:
            data = file.readlines()
            data = data[0].lower().replace(" ", "").split(',')
            return data

    def remove_conjunc_adject(self):
        '''
        removes the unnecessary words from the text data
        param: text file
        return: list of words 
        '''
        res = []
        with open(self.infile, 'r') as file:
            in_data = file.readline().splitlines()
            while in_data:
                if in_data != ['']:
                    in_data = in_data[0].replace('/', ' ').lower().split()
                    for word in in_data:
                        word = self.__chr_cleaner(word)
                        if not word in self.ignore:
                            res.append(word)
                in_data = file.readline().splitlines()
        return res

    def count_words(self, data):
        '''
        counts how many times each word in a list appeared in the list passed
        returns the result in dict format (e.g., {'collaboration': 3})\
            which is sorted by counts in descnding order
        '''
        res = dict()
        for word in data:
            if word in res.keys():
                res[word] += 1
            else:
                res[word] = 1
        res = sorted(res.items(), key=lambda x: x[1], reverse=True)
        return res
    
    def finished_description(self):
        res = self.word_counted_description
        return res
    
    def __chr_cleaner(self, word):
        '''
        replaces the symbols/characters deemed unnecessary
        '''
        return ''.join([i for i in word if i not in self.REPLACE]).lower()

class ResumeCleaner(JobDescriptionCleaner):

    def __init__(self, resume_file):
        '''
        1. asks to specify the path to resume file
        2. creates a list of irrelevant words identified beforehand
        3. use the list of irrelevant words and the list of conjunctions\
            to clean the resume
        4. count how many times each word in resume appeared 
        '''
        super().__init__(resume_file)
        self.resume = resume_file
        self.remove_words = self.__unnecessaries_list()
        self.cleaned_resume = self.remove_conjunc_adject()
        self.very_clean_resume = self.finalized_resume()
        self.word_counted_resume = self.count_words(self.very_clean_resume)


    def finalized_resume(self):
        '''
        param: list of words found in resume, AFTER the conjunction words are removed
        remove irrelevant words from the resume
        '''
        res = []
        for word in self.cleaned_resume:
            word = word.lower()
            if not word in self.remove_words:
                res.append(word)
        return res
    
    def finished_resume(self):
        return self.word_counted_resume

    def __unnecessaries_list(self):
        '''
        read the file IRRELEVANT and turns the content into a list
        '''
        file_path = PATH+self.IRRELEVANT
        with open(file_path, 'r') as file:
            words = file.readlines()
            words = words[0].lower().split(',')
            return words

class DocComparision(ResumeCleaner):
    '''
    the primary objectives of this class are:\
        1. create a pd.DataFrame of all words appered in job description and resume\
            and the counts
        2. use sklearn package to compares the two lists of words\
            were extracted from resume and job description\
                calculates a number to represent the extent\
                    to which the two data under analysys\
                        resemble one another
    '''

    def __init__(self, resume_file, job_description):
        '''
        1. convert job description words & counts in tupel back into a key-value pair object
        2. convert resume words & counts in tupel back into a key-value pair object 
        3. consolidates the two key-value pair objects created in above into one\
            that can be processed by pandas
        4. create a pd.DataFrame with consolidated data
        5. converts each list of words (resume and job description)\
            into a long string of words, so that they can be processed for similarities\
                by sklearn
        6. calculate how similar the two data are
        7. ask user to specify the file path to write all the result to
        8. writes the result
        '''
        self.job_description_filtered_listed = JobDescriptionCleaner(job_description).remove_conjunc_adject()
        self.job_description_finalized = JobDescriptionCleaner(job_description).finished_description()
        self.resume_filtered_listed = ResumeCleaner(resume_file).finalized_resume()
        self.resume_file_finalized = ResumeCleaner(resume_file).finished_resume()

        self.employers_words = self.__tup_to_dict(self.job_description_finalized)
        self.my_words = self.__tup_to_dict(self.resume_file_finalized)
        self.processed_data = self.__check_counts(self.employers_words, self.my_words)
        self.df = self.to_dataframe(self.processed_data)
        self.resume_text = ' '.join(self.resume_filtered_listed)
        self.description_text = ' '.join(self.job_description_filtered_listed)
        self.match_score = self.__match_score(self.description_text, self.resume_text)
        

    def to_dataframe(self, data_dict):
        '''
        create pd.DataFrame from consolidated dict object
        '''
        df = pd.DataFrame.from_dict({(k): data_dict[k] for k in data_dict.keys() for v in data_dict[k].keys()}, orient='index')
        df.sort_values(by='employer', inplace=True, ascending=False)
        return df

    def comparison_result(self):
        res = self.__finalized_data()
        return res


    def __check_counts(self, employers_dict, my_dict):
        '''
        Compares dict_1 and dict_2 and writes a new dict
        param: 2 dictionaries
        return: the new dict (keys == words
        \ vals == {'employer: num, 'me': num})
        '''
        res = dict()
        emp_words = employers_dict.keys()
        my_words = my_dict.keys()
        for word in my_dict.keys():
            if word in emp_words:
                res[word] = {'employer': employers_dict[word], 'me': my_dict[word]}
            else:
                res[word] = {'employer': 0, 'me': my_dict[word]}
        
        for word in employers_dict.keys():
            if not word in my_words:
                res[word] = {'employer': employers_dict[word], 'me': 0}
        
        return res
                
    def __tup_to_dict(self, tup_lst):
        '''
        turns tutple into dict object
        '''
        res = dict()
        for tup in tup_lst:
            res[tup[0]] = tup[1]
        return res

    def __match_score(self, description, resume):
        '''
        calculate similarity between two objects
        '''
        description_resume_text = [description, resume]
        cv = CountVectorizer()
        comparison_matrix = cv.fit_transform(description_resume_text)
        res = round(cosine_similarity(comparison_matrix)[0][1]*100, 4)
        return res

    def __finalized_data(self):
        '''
        returns a dict that includes similarity score in percentage and all data in the pd.DataFrame
        '''
        score = f"The similarity between resume and job description is: {self.match_score}%"
        # df = self.df.to_string(header=True, index=True)
        df = self.df.to_html()
        res = {"score": score, "data": df}
        
        return res



if __name__ == "__main__":
    EmpAnalysis.__init__()
    HealthCarePortal.__init__()
                




