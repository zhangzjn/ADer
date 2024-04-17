
class Registry:

	def __init__(self, name):
		self.name = name
		self.name_to_fn = dict()

	def register_module(self, fn, name=None):
		module_name = name if name else fn.__name__
		self.name_to_fn[module_name] = fn
		return fn
		
	def __len__(self):
		return len(self.name_to_fn)

	def __contains__(self, name):
		return name in self.name_to_fn.keys()

	def get_module(self, name):
		if self.__contains__(name):
			return self.name_to_fn[name]
		else:
			raise ValueError('invalid module: {}'.format(name))
