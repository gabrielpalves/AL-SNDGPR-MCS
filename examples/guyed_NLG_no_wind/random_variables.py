import numpy as np


# Wind speed
# v0 = 34.94
v0 = 38

beta = v0/10.9
mu = 7*beta

gamma_e = 0.577215664901532860606512
std = beta*np.pi/np.sqrt(6)
mean = mu + beta*gamma_e
cv = std/mean

# Converts to 10 min
mean = mean*0.72
std = mean*cv

# in 'moments': [mean, CoV], inform mean and coefficient of variation
RVs = [
    # {'name': 'Wind speed',      'distribution': 'gumbel',    'moments': [mean, cv]},         # 1
    # {'name': 'f_y g50',         'distribution': 'normal',    'moments': [1.22, 0.08]},       # 2
    # {'name': 'f_y g50',         'distribution': 'lognormal', 'moments': [1.16246, 0.0418]},  # 2
    {'name': 'f_y g50',         'distribution': 'rayleigh', 'moments': [1.0695263114519873, 0.0742498104148535]},  # 2
    # {'name': 'f_y g60',         'distribution': 'normal',    'moments': [1.22, 0.08]},       # 3
    # {'name': 'f_y g60',         'distribution': 'lognormal', 'moments': [1.16246, 0.0418]},  # 3
    {'name': 'f_y g60',         'distribution': 'rayleigh', 'moments': [1.0695263114519873, 0.0742498104148535]},  # 3
    {'name': 'E',               'distribution': 'lognormal', 'moments': [1, 0.03]},          # 4
    {'name': 'E_{guys}',        'distribution': 'lognormal', 'moments': [1, 0.03]},          # 5
    {'name': 'A_{guys}',        'distribution': 'normal',    'moments': [1.03, 0.01]},       # 6
    # {'name': 'f_{u_{guys}}',    'distribution': 'normal',    'moments': [1.07, 0.015]},      # 7
    {'name': 'f_{u_{guys}}',    'distribution': 'gumbel',    'moments': [1.1864, 0.0437]},   # 7
    {'name': 'Pre-tension',     'distribution': 'gamma',     'moments': [1, 0.20]},          # 8
    {'name': 'w_{40}',          'distribution': 'normal',    'moments': [1.00075, 0.005]},   # 9
    {'name': 'w_{45}',          'distribution': 'normal',    'moments': [0.99922, 0.005]},   # 10
    {'name': 'w_{50}',          'distribution': 'normal',    'moments': [1.01800, 0.005]},   # 11
    {'name': 'w_{60}',          'distribution': 'normal',    'moments': [0.99030, 0.009]},   # 12
    {'name': 'w_{65}',          'distribution': 'normal',    'moments': [0.99030, 0.009]},   # 13
    {'name': 'w_{75}',          'distribution': 'normal',    'moments': [1.00440, 0.009]},   # 14
    {'name': 't_{3mm}',         'distribution': 'normal',    'moments': [1.0433, 0.0295]},   # 15
    {'name': 't_{4mm}',         'distribution': 'normal',    'moments': [1.0350, 0.029]},    # 16
    {'name': 't_{5mm}',         'distribution': 'normal',    'moments': [1.01, 0.0098]},     # 17
    {'name': 't_{6mm}',         'distribution': 'normal',    'moments': [1.0183, 0.022]},    # 18
    {'name': 'v',               'distribution': 'lognormal', 'moments': [1, 0.03]},          # 19
    {'name': 'd_c',             'distribution': 'normal',    'moments': [1, 0.01]},          # 20
    {'name': 'd_{gw}',          'distribution': 'normal',    'moments': [1, 0.01]},          # 21
    # {'name': 'f_u g50',         'distribution': 'gumbel',    'moments': [1.1864, 0.0437]},   # 22
    # {'name': 'f_u g60',         'distribution': 'gumbel',    'moments': [1.1864, 0.0437]},   # 23
    # {'name': 'f_ub',            'distribution': 'normal',    'moments': [1.21, 0.0413]},     # 24
    # {'name': 'A',               'distribution': 'normal',    'moments': [1, 0.129],  'bounds': [1-0.129*4,  1+0.129*4]},   # 25
    # {'name': 'B',               'distribution': 'normal',    'moments': [1, 0.0855], 'bounds': [1-0.0855*4, 1+0.0855*4]},  # 26
    # {'name': 'x',               'distribution': 'normal',    'moments': [1, 0.2001], 'bounds': [1-0.2001*4, 1+0.2001*4]},  # 27
    # {'name': 'P',               'distribution': 'normal',    'moments': [1, 0.2234], 'bounds': [1-0.2234*4, 1+0.2234*4]},  # 28
    # {'name': 'Q',               'distribution': 'normal',    'moments': [1, 0.186],  'bounds': [1-0.186*4,  1+0.186*4]},   # 29
    # {'name': 'R',               'distribution': 'normal',    'moments': [1, 0.201],  'bounds': [1-0.201*4,  1+0.201*4]},   # 30
]
