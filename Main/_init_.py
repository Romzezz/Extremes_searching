def find_local_extremas(varnames, func, bounds=None):
    """
    varnames: ['x', 'y']
    func: 'y*(x**2) + x*(y**3) - x*y'
    bounds: {'x':[-10, 10], 'y': [-1, 1]} 
    """
    assert isinstance(varnames, list), 'список переменных задан неверно'
    assert isinstance(func, str), 'функция задана неверно'
    for symbol in func:
        if symbol.isalpha() and symbol not in varnames:
            raise Exception('аргументы функции не совпадают с заданными переменными')
    
    
    args = sympy.symbols(varnames)
    sympifyed = sympy.sympify(func)
    partial_first = [sympifyed.diff(arg) for arg in args]
    stationary_points = sympy.solve(partial_first, args, dict=True)
    
    if bounds:
        assert isinstance(bounds, dict), 'ограничения заданы неверно'
        stationary_points = _filter_points(stationary_points, bounds)
        
    if not stationary_points:
        return 'Нет стационарных точек'
    
    hesse = _get_hessian(sympifyed, args)
    _show_extrema(sympifyed, stationary_points)
    return [(point, _check_point(hesse, point)) for point in stationary_points]
