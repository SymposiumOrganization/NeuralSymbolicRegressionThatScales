import setuptools 

setuptools.setup(
      version='0.1.0',
      description='Code for reproducing the results of Neural Symbolic Regression that scales',
      name="nesymres",
      packages=setuptools.find_packages('.'),
      package_dir={'': '.'},
      install_requires=[
          'numpy','sympy', 'pandas','click', 'tqdm','numexpr','jsons'
      ]
     )