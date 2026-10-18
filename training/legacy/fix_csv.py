import re

with open("etiquetado/dl/etiquetado_dl_front.csv", "rb") as f:
    raw = f.read()

text = raw.decode("utf-8-sig", errors="replace")

lines = text.strip().split("\n")
print(f"Total lineas: {len(lines)}")

header = lines[0]
print(f"Header: {header}")

# Parse each line manually
parsed = [header]
for i in range(1, len(lines)):
    line = lines[i].strip()
    if not line:
        continue

    # Remove leading/trailing garbage
    # Pattern: line starts with "dl_NUM,... and has trailing commas
    # Find the first proper field value
    line = line.strip(",")

    # If line starts with " and contains doubled quotes, clean it
    if line.startswith('"') and '""' in line:
        # Remove the outer quote
        inner = line[1:]
        # Remove trailing commas
        inner = inner.rstrip(",")
        # Remove the trailing quote if present
        if inner.endswith('"'):
            inner = inner[:-1]
        # Replace doubled quotes with single quotes
        inner = inner.replace('""', '"')
        parsed.append(inner)
    else:
        parsed.append(line)

out_path = "etiquetado/dl/etiquetado_dl_front_fixed.csv"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(parsed) + "\n")

print(f"Escritas {len(parsed)-1} lineas de datos")

# Verify
import pandas as pd
df = pd.read_csv(out_path, encoding="utf-8")
print(f"Filas: {len(df)}")
print(f"Columnas: {list(df.columns)}")
print(f"Labels: {df['label'].value_counts().to_dict()}")
print(f"Primeros 3:")
print(df[["video_id", "resultado", "label"]].head(3).to_string())
