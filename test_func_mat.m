function y = test_func_mat(a, para)

    k = para(1);
    b = para(2);
    
    y = exp(k*log(a) + b);

end