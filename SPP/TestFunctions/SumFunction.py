#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .TestFunctions import ConvFunc_OneArg,  ConvConcFunc
from .TestFunctions import TestFunction
import numpy as np
import math


class r(ConvFunc_OneArg):
	def __init__(self, size_domain = 10):
		self.L, self.M = 2, 10
		self.mu = 1
	def get_L(self):
		return self.L
		
	def get_value(self, x):
		return np.dot(x,x)
	
	def grad(self, x):
		return 2 * x

class h(ConvFunc_OneArg):
	def __init__(self):
		self.L, self.M = 2, 10
		self.mu = 1
		
	def get_value(self, y):
		return 4*np.dot(y,y) + 10*(np.sin(y[0]))**2
	
	def grad(self, y):
		vec = 8*y
		vec[0] = vec[0] + math.sin(y[0])*math.cos(y[0])
		return vec
	
class F(ConvConcFunc):
	def __init__(self, alpha = 1):
		self.L_xx, self.L_yy = 0, 0
		self.L_yx, self.L_xy = 2, 2
		self.mu_y, self.mu_x = 1, 1
		self.M_x, self.M_y = 2, 2
		self.alpha = alpha
	def get_value(self, x, y):
		return 3 * (math.sin(x[0])*math.sin(y[0]))**2

	def grad_y(self, x, y):
		vec_y = [0 for i in range(len(y))]
		vec_y[0] = 6*math.sin(y[0])*(math.sin(x[0]))**2
		return vec_y
	
	def grad_x(self, x, y):
		vec_x = [0 for i in range(len(x))]
		vec_x[0] = 6 * math.sin(x[0]) * (math.sin(y[0])) ** 2
		return vec_x

def get_test_func(alpha = 1, solver = None, get_start_point = None):
	return TestFunction(r(), F(alpha = alpha), h(), solver, get_start_point)

if __name__ == "__main__":
	TrivialFunc = get_test_func()
	print(TrivialFunc.get_value(np.ones(10,), np.ones(10,)))
	print(TrivialFunc.grad_x(np.ones(10,), np.ones(10,)))
	print(TrivialFunc.grad_y(np.ones(10,), np.ones(10,)))