#/usr/bin/bash

for i in {1..200}; do
    pytest -q -s test_loss_functions.py
done
