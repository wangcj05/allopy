import sys

from setuptools import find_packages, setup

import versioneer

PACKAGE_NAME = 'allopy'

cmdclass = versioneer.get_cmdclass()


def run_setup():
    major, minor, *_ = sys.version_info

    if major != 3:
        raise RuntimeError('Please build on python 3!')

    install_requires = [
        'copulae >=0.2',
        'numpy',
        'nlopt >=2.4',
        'scipy >=1.1',
        'pandas >=0.23'
    ]

    setup(
        name=PACKAGE_NAME,
        license='MIT',
        version=versioneer.get_version(),
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
        python_requires='>=3.6',
        zip_safe=False
    )


if __name__ == '__main__':
    run_setup()
