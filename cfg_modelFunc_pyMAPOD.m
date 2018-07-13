function f = cfg_modelFunc_pyMAPOD(a, para, funcName)

    funcName =  str2func(funcName);
    f = funcName(a, para);

end