from domains import DomainND
import numpy as np
import tensorflow as tf
from utils import multimesh, flatten_and_stack, MSE, convertTensor, get_tf_model


def get_linspace(dict_):
    lin_key = "linspace"
    return [val for key, val in dict_.items() if lin_key in key][0]


class BC(DomainND):
    def __init__(self):
        self.isPeriodic = False
        self.isInit = False
        self.isNeumann = False
        self.isDirichletFunc=False

    def compile(self):
        self.input = self.create_input()

    def get_dict(self, var):
        return next(item for item in self.domain.domaindict if item["identifier"] == var)

    def get_not_dims(self, var):
        self.dicts_ = [item for item in self.domain.domaindict if item['identifier'] != var]
        return [get_linspace(dict_) for dict_ in self.dicts_]

    def create_target_input_repeat(self, var, target):
        fidelity_key = "fidelity"
        fids = []
        for dict_ in self.dicts_:
            res = [val for key, val in dict_.items() if fidelity_key in key]
            fids.append(res)
        reps = np.prod(fids)
        if type(target) is str:
            return np.repeat(self.dict_[(var + target)], reps)
        else:
            return np.repeat(target, reps)
        
class dirichletBC(BC):
    def __init__(self, domain, val, var, target):
        self.domain = domain
        self.val = val
        self.var = var
        self.target = target
        super().__init__()
        self.dicts_ = [item for item in self.domain.domaindict if item['identifier'] != self.var]
        self.dict_ = next(item for item in self.domain.domaindict if item["identifier"] == self.var)
        self.target = self.dict_[var+target]
        self.compile()
        self.isDirichlect = True

    def create_input(self):
        repeated_value = self.create_target_input_repeat(self.var, self.target)
        repeated_value = np.reshape(repeated_value, (-1, 1))
        mesh = flatten_and_stack(multimesh(self.get_not_dims(self.var)))
        mesh = np.insert(mesh, self.domain.vars.index(self.var), repeated_value.flatten(), axis=1)
        return mesh

class FunctionDirichletBC(BC):
    def __init__(self, domain, fun, var, target, func_inputs, n_values=None):
        self.domain = domain
        self.fun = fun
        self.var = var
        self.target = target
        self.func_inputs = func_inputs
        self.n_values = n_values
        # self.dicts_ = [item for item in self.domain.domaindict if item['identifier'] != self.var]
        # self.dict_ = next(item for item in self.domain.domaindict if item["identifier"] == self.var)
        # print(self.dict_)
        super().__init__()
        # self.targets = self.dict_[var+target]
        self.compile()
        self.create_target()
        self.isDirichletFunc = True
        
    def get_input_upper_lower(self, var):
        self.repeat = self.create_target_input_repeat(var, self.target)
          
    def compile(self):
        self.input = []
        for var in self.var:
            self.dicts_ = [item for item in self.domain.domaindict if item["identifier"] != var]
            self.dict_ = next(item for item in self.domain.domaindict if item["identifier"] == var)
            self.get_input_upper_lower(var)
            mesh = flatten_and_stack(multimesh(self.get_not_dims(var)))
            self.input.append(np.insert(mesh, self.domain.vars.index(var), self.repeat.flatten(), axis=1))

        if self.n_values is not None:
            self.nums = np.random.randint(0, high=len(self.input[0]), size=self.n_values)
        else:
            self.nums = np.random.randint(0, high=len(self.input[0]), size=len(self.input[0]))

        self.input = self.unroll(self.input)
        
    def unroll(self, inp):
        outer = []
        for _, lst in enumerate(inp):
            tmp = [np.reshape(vec, (-1, 1))[self.nums] for vec in lst.T]
            outer.append(np.asarray(tmp))
        return outer

    def create_input(self):
        dims = self.get_not_dims(self.var)
        #dims = [get_linspace(dim) for dim in self.vars]
        # vals = np.reshape(fun_vals, (-1, len(self.vars)))
        mesh = flatten_and_stack(multimesh(dims))
        # dim_repeat = np.repeat(0.0, len(mesh))
        dim_repeat = self.create_target_input_repeat(self.var, self.target)
        mesh = np.insert(mesh, self.domain.vars.index(self.var), dim_repeat.flatten(), axis=1)
        if self.n_values is not None:
            self.nums = np.random.randint(0, high=len(mesh), size=self.n_values)
            mesh = mesh[self.nums]
        return mesh

    def create_target(self):
        fun_vals = []
        list=[]
        # inp=
        for i, var_ in enumerate(self.func_inputs):
            arg_list = []
            if len(self.func_inputs) > 1:
                for j, var in enumerate(var_):
                    var_dict = self.get_dict(var)
                    arg_list.append(get_linspace(var_dict))
                inp = flatten_and_stack(multimesh(arg_list))
                array=np.concatenate(inp).ravel()
                list.append(array)
                fun_vals.append(self.fun(*list))
            else:
                for j, var in enumerate(var_):
                    var_dict = self.get_dict(var)
                    arg_list.append(get_linspace(var_dict))
                inp = flatten_and_stack(multimesh(arg_list))
                fun_vals.append(self.fun(*inp.T))
        
        self.val = convertTensor(np.reshape(fun_vals, (-1, 1))[self.nums])
        
class FunctionNeumannBC(BC):
    def __init__(self, domain, fun, var, target, deriv_model, func_inputs, n_values=None):
        self.n_values = n_values
        self.domain = domain
        self.fun = fun
        self.var = var
        self.target = target
        super().__init__()
        self.deriv_model = [get_tf_model(model) for model in deriv_model]
        self.isNeumann = True
        self.func_inputs = func_inputs
        self.compile()
        self.create_target()

    def get_input_upper_lower(self, var):
        self.repeat = self.create_target_input_repeat(var, self.target)

    def compile(self):
        self.input = []
        for var in self.var:
            self.dicts_ = [item for item in self.domain.domaindict if item["identifier"] != var]
            self.dict_ = next(item for item in self.domain.domaindict if item["identifier"] == var)
            self.get_input_upper_lower(var)
            mesh = flatten_and_stack(multimesh(self.get_not_dims(var)))
            self.input.append(np.insert(mesh, self.domain.vars.index(var), self.repeat.flatten(), axis=1))

        if self.n_values is not None:
            self.nums = np.random.randint(0, high=len(self.input[0]), size=self.n_values)
        else:
            self.nums = np.random.randint(0, high=len(self.input[0]), size=len(self.input[0]))

        self.input = self.unroll(self.input)
        # self.lower = self.unroll(self.lower)

    def u_x_model(self, u_model, inputs):
        return [model(u_model, *inputs) for model in self.deriv_model]

    def unroll(self, inp):
        outer = []
        for _, lst in enumerate(inp):
            tmp = [np.reshape(vec, (-1, 1))[self.nums] for vec in lst.T]
            outer.append(np.asarray(tmp))
        return outer

    def create_target(self):
        fun_vals = []
        list=[]
        # inp=
        if len(self.func_inputs) > 1:
            for i, var_ in enumerate(self.func_inputs):
                arg_list = []
                for j, var in enumerate(var_):
                    var_dict = self.get_dict(var)
                    arg_list.append(get_linspace(var_dict))
                inp = flatten_and_stack(multimesh(arg_list))
                array=np.concatenate(inp).ravel()
                list.append(array)
                fun_vals.append(self.fun(*list))
        else:
            for i, var_ in enumerate(self.func_inputs):
                for j, var in enumerate(var_):
                    var_dict = self.get_dict(var)
                    arg_list.append(get_linspace(var_dict))
                inp = flatten_and_stack(multimesh(arg_list))
                array=np.concatenate(inp).ravel()
                list.append(array)
                fun_vals.append(self.fun(*inp.T))

        self.val = convertTensor(np.reshape(fun_vals, (-1, 1))[self.nums])
        
        # print(self.val.shape,self.nums)
def get_function_out(func, var, dict_):
    linspace = get_linspace(dict_)
    return func(linspace)


class IC(BC):
    def __init__(self, domain, fun, var, n_values=None):
        self.isInit = True
        self.n_values = n_values
        self.domain = domain
        self.fun = fun
        self.vars = var
        super().__init__()
        self.isInit = True
        self.dicts_ = [item for item in self.domain.domaindict if item['identifier'] != self.domain.time_var]
        self.dict_ = next(item for item in self.domain.domaindict if item["identifier"] == self.domain.time_var)
        self.compile()
        self.create_target()

    def create_input(self):
        dims = self.get_not_dims(self.domain.time_var)
        mesh = flatten_and_stack(multimesh(dims))
        t_repeat = np.repeat(0.0, len(mesh))

        mesh = np.concatenate((mesh, np.reshape(t_repeat, (-1, 1))), axis=1)
        if self.n_values is not None:
            self.nums = np.random.randint(0, high=len(mesh), size=self.n_values)
            mesh = mesh[self.nums]
        return mesh

    def create_target(self):
        fun_vals = []
        for i, var_ in enumerate(self.vars):
            arg_list = []
            for j, var in enumerate(var_):
                var_dict = self.get_dict(var)
                arg_list.append(get_linspace(var_dict))
            inp = flatten_and_stack(multimesh(arg_list))
            fun_vals.append(self.fun[i](*inp.T))
        if self.n_values is not None:
            self.val = convertTensor(np.reshape(fun_vals, (-1, 1))[self.nums])
        else:
            self.val = convertTensor(np.reshape(fun_vals, (-1, 1)))

    def loss(self):
        return MSE(self.preds, self.val)

class periodicBC(BC):
    def __init__(self, domain, var, deriv_model, n_values=None):
        self.n_values = n_values
        self.domain = domain
        self.var = var
        super().__init__()
        self.deriv_model = [get_tf_model(model) for model in deriv_model]
        self.isPeriodic = True
        self.compile()

    def get_input_upper_lower(self, var):
        self.upper_repeat = self.create_target_input_repeat(var, self.dict_["range"][1])
        self.lower_repeat = self.create_target_input_repeat(var, self.dict_["range"][0])

    def compile(self):
        self.upper = []
        self.lower = []
        for var in self.var:
            self.dicts_ = [item for item in self.domain.domaindict if item["identifier"] != var]
            self.dict_ = next(item for item in self.domain.domaindict if item["identifier"] == var)
            self.get_input_upper_lower(var)
            mesh = flatten_and_stack(multimesh(self.get_not_dims(var)))
            self.upper.append(np.insert(mesh, self.domain.vars.index(var), self.upper_repeat.flatten(), axis=1))
            self.lower.append(np.insert(mesh, self.domain.vars.index(var), self.lower_repeat.flatten(), axis=1))

        if self.n_values is not None:
            self.nums = np.random.randint(0, high=len(self.upper[0]), size=self.n_values)
        else:
            self.nums = np.random.randint(0, high=len(self.upper[0]), size=len(self.upper[0]))

        self.upper = self.unroll(self.upper)
        self.lower = self.unroll(self.lower)

    def u_x_model(self, u_model, inputs):
        return [model(u_model, *inputs) for model in self.deriv_model]

    def unroll(self, inp):
        outer = []
        for _, lst in enumerate(inp):
            tmp = [np.reshape(vec, (-1, 1))[self.nums] for vec in lst.T]
            outer.append(np.asarray(tmp))
        return outer



