import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(description="Balance the dataset classes by grouping low-count classes together.")
    parser.add_argument('--metadata_path', type=str, default='../data/land_cover_representation/metadata.csv', help='Path to old metadata file')
    parser.add_argument('--save_path', type=str, default='../data/land_cover_representation/metadata_balanced.csv', help='Path to save the new metadatafile')
    parser.add_argument('--min_count', type=int, default=2000, help='Minimum class count under which the class is grouped into "Other"')
    parser.add_argument('--train_split', type=float, default=0.8, help='Training set split')
    parser.add_argument('--test_split', type=float, default=0.1, help='Testing set split')
    parser.add_argument('--seed', type=int, default=42, help='Set randomness seed')
    args = parser.parse_args()

    with open(args.metadata_path) as f:
        metadata = pd.read_csv(f)

    class_distribution = pd.Series(metadata['land_cover']).value_counts()
    classes_dict = pd.Series.to_dict(class_distribution)

    class_doc = dict(zip(pd.unique(metadata['y']), pd.unique(metadata['land_cover'])))
    class_doc = dict(sorted(class_doc.items()))

    class_count = {}
    for i in range(0, 61):
        class_count[i] = (class_doc[i], classes_dict[class_doc[i]])

    count_df = pd.DataFrame.from_dict(class_count, orient='index', columns=['land_cover', 'count'])

    low_counts_df = count_df[count_df['count'] < args.min_count]
    new_count_df = count_df[count_df['count'] >= args.min_count].reset_index(drop=True)
    new_count_df.loc[len(new_count_df)] = ['Other', sum(low_counts_df['count'])]

    # Map low-count classes to 'Other'
    low_count_classes = set(low_counts_df['land_cover'])
    metadata['land_cover'] = metadata['land_cover'].apply(lambda x: 'Other' if x in low_count_classes else x)

    # Create a mapping for the new 'y' values based on the index of new_count_df
    land_cover_to_y = {land_cover: idx for idx, (land_cover, _) in new_count_df.iterrows()}
    metadata['y'] = metadata['land_cover'].map(land_cover_to_y)

    # Recalculate class distribution for the new metadata
    new_class_distribution = metadata['land_cover'].value_counts()

    balanced_metadata = pd.DataFrame()
    for cls in new_class_distribution.index:
        class_data = metadata[metadata['land_cover'] == cls]
        train, temp = train_test_split(class_data, test_size=(1-args.train_split), random_state=args.seed, stratify=class_data['land_cover'])
        val, test = train_test_split(temp, test_size=(args.test_split/(1-args.train_split)), random_state=args.seed, stratify=temp['land_cover'])
        
        train['split_str'] = 'train'
        val['split_str'] = 'val'
        test['split_str'] = 'test'
        
        balanced_metadata = pd.concat([balanced_metadata, train, val, test])

    # Ensure the headers and structure match the original metadata file
    balanced_metadata = balanced_metadata[['file_name', 'y', 'split', 'split_str', 'land_cover']]
    balanced_metadata['y'] = balanced_metadata['y'].astype(int)
    balanced_metadata = balanced_metadata.set_index('file_name').reindex(metadata['file_name']).reset_index()

    # Save the new metadata to a CSV file
    balanced_metadata.to_csv(args.save_path, index=False)

    print(f"Balanced metadata file saved at {args.save_path}")

if __name__ == '__main__':
    main()