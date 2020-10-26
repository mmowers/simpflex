"""
Simple Flexible Electricity Model
"""

import pulp # See https://github.com/coin-or/pulp for instructions
from pdb import set_trace as b
import pandas as pd

#Switches to limit years, regions
years_filter = [2020]
regions_filter = ['p1','p2']

#CRF
crf = 0.1

#import data
df_tech_cost = pd.read_csv('inputs/tech_cost.csv', dtype={'class': 'str'})
df_time_map = pd.read_csv('inputs/time_map.csv')
df_load = pd.read_csv('inputs/load.csv') #First column is 'hour'. Other columns are region names
df_wind = pd.read_csv('inputs/wind.csv') #First column is 'hour'. Other columns are formatted like [tech]_[class]_[region]
df_upv = pd.read_csv('inputs/upv.csv') #Same as above

#Apply filters
if years_filter is not None:
    df_tech_cost = df_tech_cost[df_tech_cost['year'].isin(years_filter)].copy()
if regions_filter is not None:
    df_load = df_load[['hour']+regions_filter].copy()

#get hours per time period
df_time_hours = df_time_map.groupby(['time']).count()
hours = df_time_hours.to_dict()['hour']

#get all timeslices
times = df_time_map['time'].unique().tolist()

#Temporal aggregation of load
df_load = df_load.merge(df_time_map, how='left', on=['hour'], sort=False)
df_load.drop(columns=['hour'], inplace=True)
df_load = df_load.groupby(['time'], sort=False, as_index =False).mean()

#Reformat load and gather region list
df_load = pd.melt(df_load, id_vars=['time'], var_name='reg', value_name= 'MW')
regions = df_load['reg'].unique().tolist()

#TODO: Load should be by year.
#Create load dict (reg,time) and get list of regions
load = df_load.set_index(['reg','time']).to_dict()['MW']

#Temporal aggregation of resource. if this is slow then filter by regions_filter first...
df_res = pd.concat([df_wind, df_upv.drop(columns=['hour'])], axis=1)
df_res = df_res.merge(df_time_map, how='left', on=['hour'], sort=False)
df_res.drop(columns=['hour'], inplace=True)
df_res = df_res.groupby(['time'], sort=False, as_index =False).mean()
df_res = pd.melt(df_res, id_vars=['time'], var_name='splitthis', value_name= 'cf')
df_res[['tech','class','reg']] = df_res['splitthis'].str.split('_', expand=True)
df_res.drop(columns=['splitthis'], inplace=True)

#Filter df_res by regions_filter
df_res = df_res[df_res['reg'].isin(regions_filter)].copy()

#Create resource dict for capacity factors (tech, class, reg, time)
cf_res = df_res.set_index(['tech','class','reg','time']).to_dict()['cf']

#Track resource techs and all combinations of tech,class,reg
res_techs = df_res['tech'].unique().tolist()
res_techs_tcr = df_res[['tech','class','reg']].drop_duplicates().to_records(index=False).tolist()

#Adjust capital and fixed costs from kW to MW
df_tech_cost[['capcost', 'fom']] = df_tech_cost[['capcost', 'fom']]*1000

#Add fuelcost
df_tech_cost['fuelcost'] = df_tech_cost['heatrate'] * df_tech_cost['fuelprice']

#tech cost dicts (tech, class, year)
df_tech_cost.set_index(['tech','class','year'], inplace=True)
capcost = df_tech_cost['capcost'].to_dict()
fom = df_tech_cost['fom'].to_dict()
vom = df_tech_cost['vom'].to_dict()
fuelcost = df_tech_cost['fuelcost'].to_dict()

#Create set of valid tech, class, region, year
tcy = list(capcost.keys())
tcry = [(t,c,r,y) for (t,c,y) in tcy for r in regions if t not in res_techs or (t,c,r) in res_techs_tcr]
ryh = [(r,y,h) for r in regions for y in years_filter for h in times]
tcryh = [(t,c,r,y,h) for (t,c,r,y) in tcry for h in times]

#Create problem
prob = pulp.LpProblem("elec", pulp.LpMinimize)

#Create capacity and generation variables and linking constraint
CAP = {} #TODO: CAP needs to have build year and current year.
GEN = {}
for (tech, clss, reg, year) in tcry:
    CAP[(tech, clss, reg, year)] = pulp.LpVariable(name=tech+'|'+ clss +'|'+reg+'|'+str(year), lowBound=0, upBound=None)
    for time in times:
        GEN[(tech, clss, reg, year, time)] = pulp.LpVariable(name=tech+'|'+ clss +'|'+reg+'|'+str(year)+'|'+str(time), lowBound=0, upBound=None)
        #CONSTRAINT: GEN < CAP
        if tech in res_techs:
            prob += GEN[(tech, clss, reg, year, time)] == cf_res[(tech, clss, reg, time)] * CAP[(tech, clss, reg, year)]
        else:
            prob += GEN[(tech, clss, reg, year, time)] <= CAP[(tech, clss, reg, year)]

#Add load and reserve margin requirements
for (reg, year, time) in ryh:
    #TODO: maybe remove GEN for res_techs and use CAP*cf instead?
    #TODO: Add transmission inflows and outflows
    prob += pulp.lpSum([GEN[(tech, clss, reg, year, time)] for (tech, clss, reg, year, time) in tcryh]) >= load[(reg, time)]
    prob += pulp.lpSum([CAP[(tech, clss, reg, year)] for (tech, clss, reg, year) in tcry if tech not in res_techs]) +\
        pulp.lpSum([cf_res[(tech, clss, reg, time)] * CAP[(tech, clss, reg, year)] for (tech, clss, reg, year) in tcry if tech in res_techs]) >= 1.15 * load[(reg, time)]

#TODO:
#Resource constraints
#Transmission constraints
#Relate capacity from previous years to this year

#Objective
prob += pulp.lpSum([(vom[(tech, clss, year)] + fuelcost[(tech, clss, year)])*hours[time]/crf * GEN[(tech, clss, reg, year, time)] for (tech, clss, reg, year, time) in tcryh]) +\
        pulp.lpSum([(capcost[(tech, clss, year)] + fom[(tech, clss, year)]/crf) * CAP[(tech, clss, reg, year)] for (tech, clss, reg, year) in tcry])

#Solve
status = prob.solve()

#Show chosen variables:
gen_chosen = {(t,c,r,y,h): pulp.value(GEN[(t,c,r,y,h)]) for (t,c,r,y,h) in tcryh if pulp.value(GEN[(t,c,r,y,h)]) != 0}
cap_chosen = {(t,c,r,y): pulp.value(CAP[(t,c,r,y)]) for (t,c,r,y) in tcry if pulp.value(CAP[(t,c,r,y)]) != 0}

b()