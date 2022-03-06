#!/usr/bin/env python
# coding: utf-8

# In[13]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import sympy
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


# In[14]:


def _get_hessian(func, args):
    return sympy.Matrix([[func.diff(arg1, arg2) for arg1 in args] for arg2 in args])


def _take_input(ask_restriction=False):
    result = dict()
    result['varnames'] = list(input(f"Введите названия переменных\nПрим. x y: ").split())
    result['func'] = input('Введите функцию\nПрим. x**2 + y**2: ')
    is_bounded = int(input('Есть ли ограничения? 1-да, 0-нет: '))
    bounds = None
    if is_bounded:
        bounds = dict()
        for var in result['varnames']:
            bounds[var] = list(map(int, input(f'Введите допустимые интервалы по {var}: ').split()))
            
    if ask_restriction:
        result['restriction'] = input('Введите ограничивающую функцию: ')
        
    result['bounds'] = bounds
            
    return result


def _check_point(hesse, point):
    substituted = hesse.subs(point)
    minor_dets = [substituted[:i, :i].det() for i in range(1, hesse.shape[0] + 1)]
    if all(det > 0 for det in minor_dets):
        return 'min'
    elif all(det < 0 for det in minor_dets[::2]) and all(det > 0 for det in minor_dets[1::2]):
        return 'max'
    elif minor_dets[-1] != 0:
        return 'saddle'
    else:
        return 'required additional research'
    
    
def _filter_points(args, points, bounds):
    suitable = []
    for point in points:
        is_suitable = True
        for arg in args:
            if not(bounds[arg.name][0] <= point[arg] <= bounds[arg.name][1]):
                is_suitable = False
        if is_suitable:
            suitable.append(point)
    return suitable
    
    
def _plot(func, points, bounds=None, restriction=None):
    args = list(func.free_symbols)
    arg1, arg2 = args
    if bounds:
        x, y = list(bounds.values())
        x = np.linspace(x[0], x[1], 100)
        y = np.linspace(y[0], y[1], 100)
    else:
        if points:
            x_min = min([point[arg1] for point in points]) - 2
            x_max = max([point[arg1] for point in points]) + 2
            y_min = min([point[arg2] for point in points]) - 2
            y_max = max([point[arg2] for point in points]) + 2
        else:
            x_min, x_max = -5, 5
            y_min, y_max = -5, 5
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        
    x, y = np.meshgrid(x, y)
    z = sympy.lambdify(args, func)(x, y)
    scatters = []
    if points:
        colors = {
            'min': 'red',
            'max': 'green',
            'saddle': 'yellow'
        }
        for type_ in ('min', 'max', 'saddle'):
            required = [point for point in points if point['type']==type_]
            if not required:
                continue
            x_ = [point[arg1] for point in required]
            y_ = [point[arg2] for point in required]
            z_ = [point['F'] for point in required]
            curr_scatter = go.Scatter3d(x=x_, y=y_, z=z_,
                                        surfacecolor=colors[type_], mode='markers',
                                        marker=dict(color=colors[type_]), name=type_, showlegend=True)
            scatters.append(curr_scatter)
    surface = go.Surface(z=z, x=x, y=y, opacity=0.5, colorscale='inferno')
    figure = [surface] + scatters
    if restriction:
        restriction_z = sympy.lambdify(args, restriction)(x, y)
        figure.append(go.Surface(z=restriction_z, x=x, y=y, opacity=0.5, showscale=False, colorscale='ice'))
    fig = go.Figure(data=figure)
    fig.update_layout(title='3-D график функции с отмеченными точками локальных экстремумов.',
                      scene=dict(
                          xaxis_title=arg1.name,
                          yaxis_title=arg2.name,
                          zaxis_title=f'F({arg1}, {arg2})'),
                      legend=dict(x=0)
                      )
    fig.show()


# In[15]:


def find_local_extremas():
    """
    varnames: ['x', 'y']
    func: 'y*(x**2) + x*(y**3) - x*y'
    bounds: {'x':[-10, 10], 'y': [-1, 1]} 
    """
    
    _input = _take_input()
    varnames = _input['varnames']
    func = _input['func']
    assert isinstance(varnames, list), 'список переменных задан неверно'
    assert isinstance(func, str), 'функция задана неверно'
    args = sympy.symbols(varnames)
    sympifyed = sympy.sympify(func)
    bounds = _input['bounds']
    partial_first = [sympifyed.diff(arg) for arg in args]
    stationary_points = sympy.solve(partial_first, args, dict=True)
    
    if bounds:
        assert isinstance(bounds, dict), 'ограничения заданы неверно'
        stationary_points = _filter_points(args, stationary_points, bounds)
        
    if not stationary_points:
        print('Нет стационарных точек')
    
    hesse = _get_hessian(sympifyed, args)
    points_to_remove = []
    for point in list(stationary_points):
        contains_complex = False
        for arg in args:
            if point[arg].has(sympy.I):
                contains_complex = True
        if contains_complex:
            points_to_remove.append(point)
        else:
            for arg in args:
                point[arg] = float(point[arg])
                point['F'] = float(sympifyed.subs(point))
                point['type'] = _check_point(hesse, point)
    for point in points_to_remove:
        stationary_points.remove(point)
        
    _plot(sympifyed, stationary_points, bounds=bounds)
    return stationary_points


def lagrange():
    """
    varnames: ['x', 'y']
    func: 'x*y'
    restriction: 'x**2 + 4*y**2 - 1'
    bounds: {'x':[-10, 10], 'y': [-1, 1]} 
    """
    
    _input = _take_input(1)
    varnames = _input['varnames']
    assert isinstance(varnames, list), 'список переменных задан неверно'
    args = sympy.symbols(varnames)
    func = _input['func']
    assert isinstance(func, str), 'функция задана неверно'
    func = sympy.sympify(func)
    bounds = _input['bounds']
    restriction = _input['restriction']
    assert isinstance(restriction, str), 'ограничивающая функция задана неверно'
    restriction = sympy.sympify(restriction)
    
    lambda_ = sympy.Symbol('lambda')
    lagrangian = func + lambda_*restriction
    partial_first = [lagrangian.diff(arg) for arg in args+[lambda_]]
    stationary_points = sympy.solve(partial_first, args+[lambda_], dict=True)
        
    
    if bounds:
        assert isinstance(bounds, dict), 'ограничения заданы неверно'
        stationary_points = _filter_points(args, stationary_points, bounds)
     
    if not stationary_points:
        print('Нет стационарных точек')
    
    hesse = _get_hessian(lagrangian, args)
    for point in stationary_points:
        for arg in args:
            point[arg] = float(point[arg])
        point['F'] = float(func.subs(point))
        point['type'] = _check_point(hesse, point)
    
    _plot(func, stationary_points, restriction=restriction, bounds=bounds)
    return stationary_points


# In[16]:


lagrange()


# In[ ]:




