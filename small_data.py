from ase.db import connect

#Connect
data = connect("/Data2/jstoppelman/codes/MLCouplingNew/qm_orbitals/nap_ant_combo/naphthalene_overlap.db")
print(len(data))

with connect("naphthalene_overlap_100.db", append=False) as out:
    for row in data.select(limit=100):
        out.write(row, key_value_pairs=row.key_value_pairs, data=row.data)

data2 = connect('naphthalene_overlap_100.db')

n = len(data)
assert n % 2 == 0, f"odd number of rows: {n}"

rows = data2.select(sort="id")
with connect("naphthalene_overlap_pairs.db", append=False) as out:
    for r1, r2 in zip(rows, rows):          # yields (row 0, row 1), (row 2, row 3), ...
        atoms = r1.toatoms() + r2.toatoms()
        kvp = {f"{k}_1": v for k, v in r1.key_value_pairs.items()}
        kvp.update({f"{k}_2": v for k, v in r2.key_value_pairs.items()})
        d = {f"{k}_1": v for k, v in r1.data.items()}
        d.update({f"{k}_2": v for k, v in r2.data.items()})
        out.write(atoms, key_value_pairs=kvp, data=d)

print(len(out))  # n / 2
