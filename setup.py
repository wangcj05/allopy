from setuptools import find_packages, setup

import versioneer

PACKAGE_NAME = 'allopy'
VERSION = versioneer.get_version().split('+')[0]

cmdclass = versioneer.get_cmdclass()

install_requires = [
    'copulae >=0.4',
    'muarch',
    'numpy',
    'nlopt >=2.6',
    'scipy',
    'pandas'
]

setup(
    name=PACKAGE_NAME,
    license='MIT',
    version=VERSION,
    description='Toolbox for portfolio construction and financial engineering',
    author='Daniel Bok',
    author_email='daniel.bok@outlook.com',
    packages=find_packages(include=['allopy', 'allopy.*']),
    url='https://github.com/DanielBok/allopy',
    cmdclass=cmdclass,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    install_requires=install_requires,
    include_package_data=True,
    python_requires='>=3.6',
    zip_safe=False
)
