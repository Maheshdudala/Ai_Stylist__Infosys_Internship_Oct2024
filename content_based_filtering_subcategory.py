

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('Final Fashion Dataset.csv')


catnames = [ 'gender', 'masterCategory', 'subCategory', 'articleType',
    'baseColour', 'season','usage','Month']
numnames = ['id','year','ratings','Price (USD)']

for i in catnames:
    print(f'{i}: {df[i].unique()}')

dict_unique = {
    "numerical_columns": numnames,
    "unique_values": [df[i].nunique() for i in numnames]
}

df_unique = pd.DataFrame(dict_unique)
df_unique










from sklearn.preprocessing import LabelEncoder,StandardScaler
le = LabelEncoder()
df_encoded = df.copy()
for i in catnames:
    df_encoded[i] = le.fit_transform(df[i])



features_to_be_scaled = ['gender', 'masterCategory', 'subCategory', 'articleType',
    'baseColour', 'season', 'ratings', 'Price (USD)', 'Month', 'year','usage']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded[features_to_be_scaled])



from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

from sklearn.cluster import KMeans
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k,random_state=42)
    kmeans.fit(pca_result)
    wcss.append(kmeans.inertia_)


df_pca = pd.DataFrame(pca_result,columns=['pca1','pca2'])

kmeans = KMeans(n_clusters=4,random_state=42)
df_pca["cluster"] = kmeans.fit_predict(df_pca)
df_pca

df_pca.isnull().sum()

df.reset_index(drop=True, inplace=True)
df_pca.reset_index(drop=True, inplace=True)

df['cluster'] = df_pca['cluster']

df.isnull().sum()


"""***Content-based filtering***"""

df

df.columns

numnames

df.isnull().sum()

df_encoded = df.copy()
le_dict = {}

for col in catnames:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    le_dict[col] = le

print("Encoded DataFrame:")
print(df_encoded)

subcategory_mapping = {label: idx for label, idx in zip(le_dict['subCategory'].classes_, le_dict['subCategory'].transform(le_dict['subCategory'].classes_))}
decode_mapping = {idx: label for idx, label in zip(le_dict['subCategory'].transform(le_dict['subCategory'].classes_),le_dict['subCategory'].classes_)}
gender_mapping = {label: idx for label, idx in zip(le_dict['gender'].classes_, le_dict['gender'].transform(le_dict['gender'].classes_))}
colour_mapping = {label: idx for label, idx in zip(le_dict['baseColour'].classes_, le_dict['baseColour'].transform(le_dict['baseColour'].classes_))}

print("Subcategory Mapping (Original to Encoded):", subcategory_mapping)
print("Decoded Subcategory Mapping (Encoded to Original):", decode_mapping)
print("Gender Mapping (Original to Encoded):", gender_mapping)
print("Colour Mapping (Original to Encoded):", colour_mapping)

df_encoded.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)

df_encoded['cluster'] = df['cluster']


df_encoded.isnull().sum()

# Scale numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_encoded[['ratings', 'Price (USD)','year']])
scaled_df = pd.DataFrame(scaled_features, columns=['ratings', 'Price (USD)','year'])

processed_dataset = pd.concat([df_encoded[['id', 'gender', 'masterCategory', 'subCategory', 'articleType','baseColour', 'season', 'usage', 'productDisplayName', 'Month','cluster']],scaled_df], axis=1)
processed_dataset



processed_dataset['combined_features'] = processed_dataset[['gender', 'baseColour', 'masterCategory',
                                             'subCategory', 'articleType', 'year',
                                             'Month', 'ratings', 'season',
                                             'usage', 'Price (USD)', 'productDisplayName']].astype(str).agg(' '.join, axis=1)


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_dataset['combined_features'])

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

from sklearn.metrics.pairwise import cosine_similarity

similarity_matrices = {}

for cluster_id in range(4):
    cluster_items = processed_dataset[processed_dataset['cluster'] == cluster_id]
    cluster_tfidf = tfidf_matrix[cluster_items.index]

    similarity_matrix = cosine_similarity(cluster_tfidf)
    similarity_matrices[cluster_id] = similarity_matrix

complementary_map = {
    'Topwear': ['Bottomwear', 'Shoes', 'Flip Flops', 'Jewellery', 'Eyewear', 'Belts', 'Bags', 'Watches', 'Wallets'],
    'Bottomwear': ['Topwear', 'Shoes', 'Flip Flops', 'Bags', 'Belts', 'Watches', 'Wallets'],
    'Bags': ['Topwear', 'Bottomwear', 'Shoes', 'Watches', 'Jewellery', 'Eyewear'],
    'Watches': ['Topwear', 'Bottomwear', 'Bags', 'Shoes', 'Jewellery'],
    'Shoes': ['Topwear', 'Bottomwear', 'Flip Flops', 'Socks', 'Eyewear', 'Belts'],
    'Flip Flops': ['Topwear', 'Bottomwear', 'Socks', 'Belts', 'Eyewear'],
    'Jewellery': ['Topwear', 'Saree', 'Watches', 'Eyewear'],
    'Eyewear': ['Topwear', 'Bottomwear', 'Shoes', 'Jewellery'],
    'Belts': ['Topwear', 'Bottomwear', 'Shoes', 'Flip Flops', 'Watches'],
    'Saree': ['Jewellery', 'Bags', 'Shoes'],
    'Loungewear and Nightwear': ['Bottomwear', 'Socks'],
    'Wallets': ['Topwear', 'Bottomwear', 'Bags', 'Watches'],
    'Socks': ['Shoes', 'Flip Flops', 'Loungewear and Nightwear']
}
encoded_complementary_map = {
    subcategory_mapping[original]: [subcategory_mapping[comp] for comp in complements if comp in subcategory_mapping]
    for original, complements in complementary_map.items() if original in subcategory_mapping
}

print("Encoded Complementary Map:", encoded_complementary_map)

colour_contrast_map = {
    'Black': ['White', 'Grey', 'Beige', 'Silver', 'Cream'],
    'Grey': ['Black', 'White', 'Red', 'Navy Blue'],
    'Blue': ['White', 'Cream', 'Yellow', 'Pink', 'Silver'],
    'Pink': ['Navy Blue', 'White', 'Beige', 'Grey'],
    'Brown': ['Cream', 'White', 'Beige', 'Olive', 'Mustard'],
    'Cream': ['Black', 'Blue', 'Brown', 'Navy Blue', 'Purple'],
    'Green': ['White', 'Black', 'Navy Blue', 'Yellow', 'Brown'],
    'White': ['Black', 'Blue', 'Red', 'Navy Blue', 'Grey'],
    'Navy Blue': ['White', 'Yellow', 'Cream', 'Pink', 'Beige'],
    'Yellow': ['Black', 'Navy Blue', 'Grey', 'Green', 'Purple'],
    'Silver': ['Black', 'Blue', 'Pink', 'Red'],
    'Red': ['White', 'Black', 'Grey', 'Beige'],
    'Beige': ['Navy Blue', 'Black', 'Red', 'Olive', 'Cream'],
    'Maroon': ['Cream', 'Beige', 'White', 'Olive'],
    'Gold': ['Black', 'White', 'Cream', 'Navy Blue'],
    'Magenta': ['White', 'Navy Blue', 'Cream'],
    'Lavender': ['Navy Blue', 'White', 'Grey'],
    'Multi': ['White', 'Black'],
    'Purple': ['Yellow', 'White', 'Cream'],
    'Charcoal': ['White', 'Cream', 'Blue'],
    'Orange': ['White', 'Black', 'Navy Blue'],
    'Tan': ['Navy Blue', 'White', 'Black'],
    'Olive': ['White', 'Yellow', 'Cream', 'Tan'],
    'Off White': ['Black', 'Grey', 'Navy Blue'],
    'Grey Melange': ['Black', 'White', 'Red', 'Blue'],
    'Rust': ['White', 'Cream', 'Beige', 'Navy Blue'],
    'Turquoise Blue': ['White', 'Black', 'Yellow'],
    'Mustard': ['White', 'Black', 'Brown', 'Navy Blue'],
    'Khaki': ['White', 'Navy Blue', 'Brown'],
    'Lime Green': ['White', 'Black', 'Grey'],
    'Peach': ['Blue', 'Black', 'Navy Blue'],
    'Sea Green': ['White', 'Black', 'Navy Blue'],
    'Teal': ['White', 'Cream', 'Navy Blue'],
    'Mauve': ['White', 'Grey', 'Blue'],
    'Copper': ['White', 'Black', 'Navy Blue'],
    'Steel': ['Black', 'White', 'Beige'],
    'Bronze': ['White', 'Black'],
    'Nude': ['White', 'Black', 'Beige'],
    'Metallic': ['White', 'Black', 'Silver'],
    'Taupe': ['White', 'Black', 'Beige'],
    'Fluorescent Green': ['White', 'Black'],
    'Burgundy': ['White', 'Beige', 'Grey'],
    'Mushroom Brown': ['White', 'Grey', 'Black'],
    'Coffee Brown': ['White', 'Beige', 'Cream']
}

encoded_complementary_colour_map = {
    colour_mapping[original]: [colour_mapping[comp] for comp in complements if comp in colour_mapping]
    for original, complements in colour_contrast_map.items() if original in colour_mapping
}

print("Encoded Complementary Map:", encoded_complementary_colour_map)

len(processed_dataset[processed_dataset.cluster==2].values)

def get_recommendations(product_id, df_encoded, similarity_matrices, encoded_complementary_map, encoded_complementary_colour_map):
    """
    Generate product recommendations based on similarity and complementary attributes.

    Args:
        product_id (int): The ID of the product for which recommendations are to be generated.
        df_encoded (pd.DataFrame): DataFrame containing encoded product information, including 'id', 'cluster', 'subCategory', 'gender', and 'baseColour'.
        similarity_matrices (dict): Dictionary of similarity matrices for each cluster.
        encoded_complementary_map (dict): Dictionary mapping encoded subcategories to their complementary subcategories.
        encoded_complementary_colour_map (dict): Dictionary mapping encoded colours to their complementary colours.

    Returns:
        list: A list of recommended product IDs.
    """
    cluster_id = df_encoded.loc[df_encoded['id'] == product_id, 'cluster'].values[0]
    cluster_df = df_encoded[df_encoded['cluster'] == cluster_id].reset_index(drop=True)
    num_recommendations = len(cluster_df)
    try:
        product_index = cluster_df[cluster_df['id'] == product_id].index[0]
    except IndexError:
        return []

    encoded_subcategory = cluster_df.loc[product_index, 'subCategory']
    product_gender = cluster_df.loc[product_index, 'gender']
    encoded_colour = cluster_df.loc[product_index, 'baseColour']

    similarity_scores = similarity_matrices[cluster_id][product_index]
    similar_indices = similarity_scores.argsort()[::-1][1:num_recommendations + 1]

    recommendations = {}
    complementary_subcategories = encoded_complementary_map.get(encoded_subcategory, [])
    complementary_colours = encoded_complementary_colour_map.get(encoded_colour, [])

    for index in similar_indices:
        recommended_product = cluster_df.iloc[index]
        recommended_subcategory = recommended_product['subCategory']
        recommended_colour = recommended_product['baseColour']

        if (recommended_subcategory in complementary_subcategories and
            recommended_product['gender'] == product_gender and
            recommended_colour in complementary_colours):

            if recommended_subcategory not in recommendations:
                recommendations[recommended_subcategory] = recommended_product['id']

            if len(recommendations) == len(complementary_subcategories):
                break

    if len(recommendations) < len(complementary_subcategories):
        remaining_subcategories = set(complementary_subcategories) - set(recommendations.keys())
        for subcategory in remaining_subcategories:
            additional_products = df_encoded[(df_encoded['subCategory'] == subcategory) &
                                             (df_encoded['gender'] == product_gender) &
                                             (df_encoded['baseColour'].isin(complementary_colours))]
            if not additional_products.empty:
                recommendations[subcategory] = additional_products.iloc[0]['id']

    return list(recommendations.values())

processed_dataset.id.sample(5)


for product_id in processed_dataset['id'].sample(10, random_state=None):
    recommended_products = get_recommendations(product_id, processed_dataset, similarity_matrices, encoded_complementary_map, encoded_complementary_colour_map)
    print(f"Product ID: {product_id}")
    print("Recommended products:", recommended_products)
    path_images = []
    image_url = "images//images//"
    for i in recommended_products:
        path_images.append(image_url + df[df.id == i]['filename'].values[0])
    print("Image paths:", path_images)
    print()
recommended_products = get_recommendations(product_id,processed_dataset,similarity_matrices,encoded_complementary_map,encoded_complementary_colour_map)
print("Recommended products:", recommended_products)

from PIL import Image
path_images = []
image_url = "images//images//"
for i in recommended_products:
    path_images.append(image_url + df[df.id==i]['filename'].values[0])
path_images


rec_images = []

for i in rec_images:
    plt.imshow(i)
    plt.show()
    plt.axis("off")
    # Evaluate the recommendations
def evaluate_recommendations(df_encoded, similarity_matrices, encoded_complementary_map, encoded_complementary_colour_map, product_ids):
    correct_recommendations = 0
    total_recommendations = 0
    total_complementary_subcategories = 0
    precision_sum = 0
    recall_sum = 0

    for product_id in product_ids:
        recommended_products = get_recommendations(product_id, df_encoded, similarity_matrices, encoded_complementary_map, encoded_complementary_colour_map)
        if recommended_products:
            total_recommendations += len(recommended_products)
            original_subcategory = df_encoded.loc[df_encoded['id'] == product_id, 'subCategory'].values[0]
            complementary_subcategories = encoded_complementary_map.get(original_subcategory, [])
            total_complementary_subcategories += len(complementary_subcategories)

            correct_count = 0
            for rec_id in recommended_products:
                rec_subcategory = df_encoded.loc[df_encoded['id'] == rec_id, 'subCategory'].values[0]
                if rec_subcategory in complementary_subcategories:
                    correct_recommendations += 1
                    correct_count += 1

            precision_sum += correct_count / len(recommended_products) if recommended_products else 0
            recall_sum += correct_count / len(complementary_subcategories) if complementary_subcategories else 0

    accuracy = correct_recommendations / total_recommendations if total_recommendations > 0 else 0
    coverage = total_recommendations / total_complementary_subcategories if total_complementary_subcategories > 0 else 0
    precision = precision_sum / len(product_ids) if len(product_ids) > 0 else 0
    recall = recall_sum / len(product_ids) if len(product_ids) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Coverage: {coverage:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1_score:.2%}")
    print(f"Average number of recommendations: {total_recommendations / len(product_ids):.2f}")

# Evaluate using subcategory mapping
product_ids = processed_dataset['id'].sample(10, random_state=42).values
evaluate_recommendations(processed_dataset, similarity_matrices, encoded_complementary_map, encoded_complementary_colour_map, product_ids)

# Display the recommended images
for i in rec_images:
    plt.imshow(i)
    plt.axis("off")
    plt.show()




