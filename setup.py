from setuptools import setup, find_packages

setup(
    name='pyembeddings',  
    version='0.1.0',
    author='Anant Sinha',
    author_email='anant@shack15.com',
    description='Simple and intuitive embeddings generation, storage, and retrieval.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/shack15/PyEmbeddings',
    packages=find_packages(),
    install_requires=[
        'requests',
        'transformers',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  # Change as appropriate
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',  # Specify compatible Python versions
)