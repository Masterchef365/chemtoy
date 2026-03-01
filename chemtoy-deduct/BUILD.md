First, run:
```sh
./import.sh
```

Then, inside the resulting Docker container, run:
```sh
python3 rmg.py /mnt/input.py
python3 /mnt/export.py | tee /mnt/chem.json
```

Now you can copy the solution out:
```
cp ./kinetics/chem.json ./src/
```
