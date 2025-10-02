import csv, os, tqdm, argparse


def main(input_csv_path, outpath):
    # Open the input CSV file
    with open(input_csv_path, 'r') as f:
        reader = csv.reader(f)

        # Create output directory if it doesn't exist
        output_dir = os.path.join(outpath,'annotations')
        os.makedirs(output_dir, exist_ok=True)

        for row in tqdm.tqdm(reader):
            if len(row) != 4:
                continue

            img_name, x, y, cl = row

            try:
                x, y = float(x), float(y)
            except ValueError:
                continue  # Skip if x or y is not a float

            txt_path = os.path.join(output_dir, img_name + '.csv')
            with open(txt_path, mode='a+', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([cl, round(x, 3), round(y, 3)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process seedpoint annotations.')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to the input seedpoints CSV file.')
    parser.add_argument('--output_dir', type=str, default='/media/vicorob/Data-1/YC/field_imagery',
                        help='Directory where annotation CSVs will be saved.')

    args = parser.parse_args()
    main(args.input_csv, args.output_dir)
