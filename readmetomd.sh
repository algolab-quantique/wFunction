#! /bin/bash

jupyter nbconvert --to markdown README.ipynb 

sed -ib 's/%%capture//g' README.md

rm README.mdb