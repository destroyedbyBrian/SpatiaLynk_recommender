import re


class Cleaner:
    def __init__(self, file_name: str):
        self.file_name = file_name

    def open_and_clean(self, output_file: str):
        cleaned_rows = []

        with open(self.file_name, 'r', encoding='utf-8') as file:
            for line in file:
                columns = line.strip().split(',')

                columns[0] = re.sub(r'[^A-Za-z0-9 .,/\-()&]', '', columns[0])

                # Remove columns 4, 5, 6, 7
                for idx in sorted([4, 5, 6, 7], reverse=True):
                    if len(columns) > idx:
                        del columns[idx]

                # Remove rows with ANY missing value
                if any(col.strip() == "" for col in columns):
                    continue

                if any('+' in col for col in columns):
                    continue

                cleaned_rows.append(columns)

        # Save clean file
        with open(output_file, 'w', encoding='utf-8') as f:
            for row in cleaned_rows:
                f.write(','.join(row) + '\n')

        print(f"Cleaned file saved as {output_file}")

    def check_duplicates(self, cleaned_file: str):
        seen = set()
        duplicates = []

        with open(cleaned_file, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line in seen:
                    duplicates.append(line)
                else:
                    seen.add(line)

        if duplicates:
            print("Duplicate entries found:")
            for dup in duplicates:
                print(dup)
        else:
            print("No duplicate entries found.")

    def add_region(self, file, output_file) -> int:
        REGION_MAP = {
            "WEST": {
                "boon lay", "choa chu kang", "clementi", "jurong east", "jurong west",
                "pioneer", "tengah", "tuas", "western water catchment", "western islands"
            },
            "NORTH": {
                "mandai", "sembawang", "simpang", "sungei kadut",
                "woodlands", "yishun"
            },
            "NORTH-EAST": {
                "ang mo kio", "hougang", "punggol",
                "seletar", "sengkang", "serangoon"
            },
            "EAST": {
                "bedok", "changi", "changi bay", "pasir ris",
                "paya lebar", "tampines"
            },
            "CENTRAL": {
                "bishan", "bukit merah", "bukit timah", "downtown core",
                "geylang", "kallang", "marina east", "marina south",
                "marine parade", "museum", "newton", "novena", "orchard",
                "outram", "queenstown", "river valley", "rochor",
                "singapore river", "southern islands", "straits view",
                "tanglin", "toa payoh"
            }
        }
        with open(file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            header = next(infile).strip().split(',')
            header.append("region")
            outfile.write(','.join(header) + "\n")

            for line in infile:
                columns = line.strip().split(',')
                location = columns[4].lower() 
                region_found = False

                for region, areas in REGION_MAP.items():
                    if any(area in location for area in areas):
                        columns.append(region)
                        region_found = True
                        break

                if not region_found:
                    continue
                
                outfile.write(','.join(columns) + '\n')

if __name__ == "__main__":
    cleaner = Cleaner("POI.csv")
    # cleaner.open_and_clean("POI_cleaned.csv")
    # cleaner.check_duplicates("POI_cleaned.csv")
    # cleaner.add_region("POI_cleaned.csv", "POI_with_region.csv")
    cleaner.container_detector("POI_with_region.csv")