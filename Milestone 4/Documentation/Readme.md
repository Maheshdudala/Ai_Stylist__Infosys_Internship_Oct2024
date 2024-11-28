# AI Stylist Team 1 Documentation

## Infosys Springboard Internship 5.0
### AI Stylist
“Automate the work of a clothing personal stylist”

## Table of Contents
- List of Figures
- Abstract
- Chapter 1: Introduction
- Chapter 2: Problem Statement and Milestones
  - 2.1 Problem Statement
  - 2.2 Milestones
- Chapter 3: System Analysis and Design
  - 3.1 System Overview
  - 3.2 Software Requirement Specifications
  - 3.3 Use Case Diagram and Flow Chart
- Chapter 4: Methodology
  - 4.1 Implementation
- Chapter 5: Results
  - 5.1 Evaluation Metrics
  - 5.2 Comparative Analysis
  - 5.3 Performance Metrics
  - 5.4 Key Findings
- Chapter 6: Conclusion and Future Work
  - 6.1 Conclusion
  - 6.2 Future Work
- References

## List of Figures
- **Figure 3.1**: Use Case Diagram for User and System
- **Figure 3.2**: Flow Chart depicting the steps of Model Building and Training
- **Figure 4.1**: Dataset
- **Figure 4.2**: Missing Data Representation
- **Figure 4.3**: Before Outlier Removal
- **Figure 4.4**: After Outlier Removal
- **Figure 4.5**: Histograms Depicting Distribution of Data
- **Figure 4.6**: Oversampling “Year” Column
- **Figure 4.7**: Count Plot for Gender and Master Category
- **Figure 4.8**: Box Plot for Subcategory
- **Figure 4.9**: Encodings
- **Figure 4.10**: Scaled Data
- **Figure 4.11**: Clusters
- **Figure 5.1**: Evaluation Metrics Scores
- **Figure 5.2**: Recommendations

## Abstract
Everyone loves to style. Having a personal stylist who curates a selection of outfits to help individuals feel comfortable and great is desirable but inaccessible to most. AI Styling now auto-cares for this process by providing intelligent outfit recommendations using advanced deep learning techniques. Our AI Styling tool harnesses a hybrid DNN model that is capable of understanding fashion aesthetics and suggesting complementary outfits based on user input. This tool aims to redefine personal styling by making it more accessible and efficient through AI-driven insights. The project also involves complete Exploratory Data Analysis and advanced image processing techniques for providing accurate and meaningful recommendations. The dataset has 44,424 rows and 10 columns, which is the basis for the model building. The dataset for this project shows the growing e-commerce industry, with a wide array of data ready to be analysed. It includes high-resolution product images shot professionally and multiple label attributes describing the products, along with descriptive text that highlights the product characteristics. Each product is given a unique ID and catalogued in a structured manner within the dataset.

## Chapter 1: Introduction
This project develops an advanced Hybrid Recommendation System for clothing and accessories, utilizing a Deep Neural Network (DNN) model to integrate collaborative filtering and content-based filtering techniques. As the e-commerce industry grows, personalized shopping experiences are in high demand, and this system addresses that by combining the strengths of both approaches. Collaborative filtering learns from past user-item interactions, such as ratings and reviews, while content-based filtering recommends items based on their attributes, like category, colour, season, and price. By integrating these methods into a hybrid model, the system provides more accurate, dynamic, and personalized recommendations.

The DNN model uses embeddings for both user and item interactions, which are concatenated with content-based features, allowing the model to capture complex relationships between users, products, and their attributes. The architecture includes embedding layers for user and item features, followed by dense layers to learn non-linear relationships. Contextual features, such as user gender, purchase month, and product price, are also included to refine the recommendations further. This hybrid approach ensures that the system delivers personalized suggestions that align with both user preferences and product characteristics.

The goal is to enhance the shopping experience by offering tailored outfit recommendations, increasing user satisfaction, and improving the likelihood of purchases on e-commerce platforms. By adapting to user behaviour and product features, the model ensures that recommendations remain relevant over time, ultimately improving customer engagement and retention. This project demonstrates how combining collaborative and content-based filtering with deep learning can create a highly effective recommendation system for dynamic and personalized e-commerce experiences.

## Chapter 2: Problem Statement and Milestones

### 2.1 Problem Statement
The e-commerce industry has seen exponential growth, leading to an overwhelming amount of product data available online. Customers often face difficulties in navigating through the vast number of products, making it hard to find items that align with their preferences. To address this challenge, an effective recommendation system is crucial to enhance the customer shopping experience by providing personalized and relevant outfit suggestions. Traditional recommendation systems based on collaborative filtering or content-based filtering alone often fall short in capturing the complex interactions between users and products. This project aims to develop a hybrid recommendation system using both collaborative and content-based filtering approaches, integrated with a Deep Neural Network (DNN) model to provide accurate, personalized outfit suggestions. The objective is to create a model that learns from both user-item interactions and item attributes, delivering high-quality recommendations that are contextually relevant and reflect user preferences.

### 2.2 Milestones
- **Milestone 1 (Week 1-2): Data Collection and Exploration**
  - Collect relevant product and user data and explore the features to understand the business context.
  - Clean, augment, and visualize the data, focusing on features that represent user preferences.
  - Handle high-dimensional data, perform outlier detection, and apply clustering techniques to identify user and product patterns.
- **Milestone 2 (Week 3-5): Model Development and Integration**
  - Experiment with traditional recommendation models like content-based and collaborative filtering.
  - Implement advanced collaborative filtering methods such as Probabilistic Matrix Factorization and Singular Value Decomposition.
  - Build and train a DNN model to integrate both user-item interactions and item attributes for personalized recommendations.
- **Milestone 3 (Week 6-7): Model Evaluation and Refinement**
  - Use evaluation metrics like Recall@K, Precision@K, and F1@K to assess the performance of the recommendation system.
  - Refine the model based on evaluation results and prepare the final recommendation system for deployment.
- **Milestone 4 (Week 8): Presentation and Documentation**
  - Prepare a presentation which must include the details of the problem statement, details of the data collected, data preprocessing methods and its outcomes, model building methodology, performance metrics and recommendations based on the outcome.
  - Project document which should capture the same topics mentioned above in more detailed format.

## Chapter 3: System Analysis and Design

### 3.1 System Overview
The system provides personalized outfit recommendations by integrating collaborative filtering, content-based filtering, and a deep neural network (DNN) to enhance predictions. The primary components include:
- **Data Collection and Preprocessing**: Gather and clean data from user interactions and product attributes, applying feature engineering to represent user preferences.
- **Collaborative Filtering**: Use user and item embeddings, creating collaborative interactions by taking the dot product of the two embeddings to predict user-item interactions.
- **Content-Based Filtering**: Recommend similar items based on product attributes (category, color, season), using an embedding layer to compute similarities.
- **Hybrid Model (Deep Neural Network)**: Combine collaborative filtering and content-based features using embeddings, followed by DNN layers to predict recommendation scores.
- **Evaluation and Metrics**: Evaluate the model using Recall@K, Precision@K, and F1@K to assess recommendation quality.

### 3.2 Software Requirement Specifications
#### Hardware Requirements:
- **Processor**: Multi-core processor (Intel i5 or above)
- **Memory**: Minimum of 8GB RAM
- **Storage**: Minimum of 50GB free disk space for storing datasets, models, and logs
- **Graphics**: GPU (for deep learning model training) with at least 4GB VRAM (e.g., NVIDIA GTX/RTX recommended)
- **Network**: A stable internet connection is essential for accessing data sources, online resources, and for any collaborative work.

#### Software Requirements:
1. **Operating System**:
   - Windows 10/11 or Linux-based OS (Ubuntu 20.04 or above): Windows is user-friendly, while Linux (Ubuntu) is ideal for machine learning due to better compatibility and resource management.
2. **Python Version**:
   - Python 3.7 or higher: Required for compatibility with machine learning libraries like TensorFlow, Keras, and others, offering improved performance and features.
3. **Libraries and Frameworks**:
   - TensorFlow: Used for building and training deep learning models
Sure, I'll continue converting the rest of the document into Markdown format for your GitHub README:


## Chapter 4: Methodology

### 4.1 Implementation

#### 4.1.1 Data Collection
The AI Stylist Recommendation System initially used a dataset containing key columns like customer details, product attributes, and purchase information but faced challenges due to the lack of product images and a small dataset size. Attempts to enhance the dataset through web scraping using Python libraries like BeautifulSoup were hindered by missing image sources, website restrictions, and file size limitations in Google Sheets. To overcome these issues, a more comprehensive dataset from Kaggle was acquired, containing most required columns and a folder of product images linked via a filename column. The dataset was further enhanced by adding new columns like month, ratings, review, and price (USD), with values populated using Python scripts.

!Dataset

#### 4.1.2 Data Preprocessing
- **Loading and Initial Exploration**: Loaded the dataset, explored its structure using descriptive statistics, and visualized data distributions with histograms and scatter plots.
- **Handling Missing Values**: Imputed missing categorical values with the most frequent or "unknown" class and numerical features with the mean/median. Removed records with missing critical identifiers like image paths.

!Missing Data Representation

- **Filtering Irrelevant Data**: Dropped metadata for corrupted/unavailable images and excluded categories with fewer than 10 items.
- **Outlier Detection and Removal**: Used z-scores and IQR to identify and remove outliers in continuous variables like price and flagged anomalies in categorical data.

!Before Outlier Removal !After Outlier Removal

- **Data Distribution Analysis**: Identified imbalances in categories and colors, preparing for augmentation and sampling to address these issues.

!Histograms Depicting Distribution of Data

- **Balancing Data**: Applied SMOTE and random oversampling to address category imbalances, creating a balanced dataset for classification tasks.

!Oversampling “Year” Column

- **Categorical Feature Analysis**: Explored key features like fabric type, occasion, and pattern, identifying anomalies and underrepresented classes.

!Count Plot for Gender and Master Category !Box Plot for Subcategory

- **Encoding Categorical Data**: Used one-hot encoding for non-ordinal features and label encoding for ordinal attributes like size.

!Encodings

- **Feature Scaling**: Standardized numerical features using MinMaxScaler and StandardScaler for uniformity across attributes.

!Scaled Data

- **Dimensionality Reduction**: Applied PCA to retain 95% variance, reducing dimensionality for clustering and classification tasks.
- **Clustering Analysis**: Performed k-means clustering, determining optimal k with the elbow method and silhouette scores, successfully grouping items for personalized recommendations.

!Clusters

#### 4.1.3 Model Building and Training

##### 4.1.3.1 Content-Based Filtering
- **Methodology**: Focused on product attributes (e.g., masterCategory, subCategory, color, gender, usage, baseColour). Computed similarity between items using cosine similarity on attribute embeddings.
- **Outcome**: Recommended similar products effectively.

##### 4.1.3.2 Collaborative Filtering
- **Methodology**: Built a user-item interaction matrix to recommend products based on user preferences.
- **Challenges**: Initial dataset lacked sufficient user interaction data. Addressed this by adding simulated user interactions: Each user interacted with 6-10 random products.
- **Outcome**: Performed well after adding synthetic user data.

##### 4.1.3.3 Hybrid DNN Model
- **Objective**: Combine content-based and collaborative filtering approaches using a deep neural network (DNN).
- **Architecture**: Input layers for user embeddings and item embeddings. Hidden layers to learn interactions between users and product features.
- **Outcome**: Improved recommendation quality by leveraging both user and item features.

#### 4.1.4 Model Evaluation
The performance of the trained hybrid recommendation model in recommending complementary items. The evaluation is performed using a held-out test set and focuses on the top 5 recommendations (K=5).

#### 4.1.5 Evaluation Metrics
The following metrics are used to assess the model's performance:
- **Precision@5**: This metric measures the proportion of correctly recommended complementary items among the top 5 recommendations made by the model. It indicates the accuracy of the recommendations.
- **Recall@5**: This metric measures the proportion of correctly recommended complementary items out of all the actual complementary items that exist for a given product. It represents the model's ability to identify relevant complementary items.
- **F1@5**: This metric is the harmonic mean of Precision@5 and Recall@5, providing a balanced measure of the model's overall performance in recommending complementary items.

Evaluation Process:
The `evaluate_model_at_k` function is used to calculate Precision@5, Recall@5, and F1@5 based on the model's predictions on the test set. The evaluation is performed on a GPU to leverage its computational power for faster processing. The calculated metrics are printed to the console, providing a quantitative assessment of the model's performance.

## Chapter 5: Results

### 5.1 Evaluation Metrics

#### 5.1.1 Overview
The performance of the hybrid Deep Neural Network (DNN) model was evaluated using ranking metrics: Recall@K, Precision@K, and F1@K. These metrics are crucial for understanding the quality of recommendations, especially in scenarios where relevance and ranking play a significant role in user satisfaction.

#### 5.1.2 Evaluation Results
Using the test dataset, the following metrics were calculated for K=5:
- **Average Precision@5**: 0.9820
- **Average Recall@5**: 0.7757
- **Average F1@5**: 0.8531

!Evaluation Metrics Scores

Insights:
- A high Recall@5 indicates the model's ability to retrieve most relevant items.
- Balanced Precision@5 and Recall@5 values suggest an effective recommendation system.

Role of Ranking Metrics:
Ranking metrics directly impact user satisfaction by ensuring the most relevant items appear at the top. They guide model optimization and inform system refinements, improving the overall quality of recommendations.

### 5.2 Comparative Analysis

#### 5.2.1 Traditional Models
- **Collaborative Filtering**: Focused on user-item interactions but lacked the capability to incorporate item-specific features (e.g., gender, season, category).
- **Content-Based Filtering**: Leveraged item attributes for recommendations but struggled to generalize across diverse user preferences.
- **Performance**: Metrics were not calculated for these models, but qualitative results suggested limited scalability.

#### 5.2.2 Hybrid Deep Neural Network (DNN)
- Combined collaborative filtering and content-based filtering, integrating user-item interactions with item attributes.
- Utilized embeddings and dense layers to learn complex relationships.

### 5.3 Performance Metrics
- **Precision@K**: Improved compared to traditional models, indicating better-ranked recommendations.
- **Recall@K**: Significantly higher, showing the ability to retrieve more relevant items.
- **F1@K**: Demonstrated a balance between precision and recall.

### 5.4 Key Findings
- **Best-Performing Model**: The hybrid DNN outperformed traditional models, excelling in both ranking metrics and recommendation quality.
- **User Experience Impact**:
  - Enhanced relevance of recommendations.
  - Greater diversity in suggested complementary items.
  - Improved user satisfaction due to personalized and precise suggestions.



## Chapter 6: Conclusion and Future Work

### 6.1 Conclusion
The AI Stylist project effectively automates personal styling by combining deep neural networks (DNN) with a hybrid approach of content-based and collaborative filtering. This integration ensures precise and diverse outfit recommendations tailored to user preferences, body types, and occasions. The system enhances user experience by incorporating real-time virtual try-ons, allowing users to visualize outfits before making decisions, bridging the gap between traditional shopping and technology.

A significant focus of the project is on promoting sustainable fashion by encouraging wardrobe optimization and eco-conscious choices, addressing the growing demand for environmentally responsible solutions. By simplifying the styling process, saving time, and offering trend-aware suggestions, the AI Stylist transforms how individuals engage with fashion.

This innovation not only benefits individuals but also holds immense potential for the fashion retail and e-commerce industries, providing personalized and scalable styling solutions. The project establishes a strong foundation for advancing AI applications in fashion, paving the way for a smarter, more inclusive, and sustainable future in personal styling.

### 6.2 Future Work
- **Enhanced Virtual Try-On Experience**: Incorporating advanced 3D modeling and augmented reality (AR) will enable more realistic visualizations of clothing, simulating fabric drape and movement to improve user confidence in outfit selections.
- **Real-Time Trend Integration**: Implementing algorithms to monitor social media, fashion blogs, and online stores will allow the system to stay updated with emerging trends, ensuring recommendations remain fashionable and relevant.
- **Diversity and Inclusivity**: Expanding the dataset to include diverse cultural styles, body types, and gender-neutral options will make the AI Stylist more inclusive, catering to a broader audience with varied fashion needs.
- **Sustainability Metrics**: Adding sustainability scores to outfit recommendations will help users make eco-friendly choices, encouraging responsible consumption based on factors like material impact and brand practices.
- **Cross-Platform E-Commerce Integration**: Integrating with major e-commerce platforms will provide a seamless experience, enabling users to browse, try on, and purchase recommended items directly within the system.

## References
1. He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.S. (2017). Neural collaborative filtering. Proceedings of the 26th International Conference on World Wide Web, pp. 173-182. Introduces deep learning methods for collaborative filtering in recommender systems.
2. Aggarwal, C.C., & Zhai, C. (2012). A survey of text clustering algorithms. In Mining Text Data (pp. 77-128). Springer. Discusses various clustering techniques applicable in recommender systems.
3. Wang, H., Zhang, F., Hou, M., Xie, X., & Guo, M. (2018). Graph-based collaborative filtering: A survey. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 48(4), pp. 607-622. Explores the use of graph-based clustering for personalized recommendations.
4. Krizhevsky, A., Sutskever, I., & Hinton, G.E. (2012). ImageNet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems, 25, pp. 1097-1105. Introduced CNNs, widely used in image processing tasks.
5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. Comprehensive reference for deep learning techniques, including image processing and dimensionality reduction.
6. Ji, S., Xu, W., Yang, M., & Yu, K. (2013). 3D convolutional neural networks for human action recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(1), pp. 221-231. Application of CNNs for multidimensional data analysis, relevant for image datasets.

### Additional References
Based on the updated code that includes PCA, KMeans clustering, and TensorFlow-based image processing, here are additional references you should include:

#### Clustering and Dimensionality Reduction
- **Scikit-learn: PCA**: Principal Component Analysis (PCA) Documentation. Used for dimensionality reduction and visualization.
- **Scikit-learn: KMeans Clustering**: KMeans Clustering Documentation. Utilized for clustering data points into distinct groups.
- **Elbow Method for Optimal K**: Scikit-learn: KMeans Elbow Method. A technique to determine the optimal number of clusters.

#### Image Processing with TensorFlow
- **TensorFlow: Image Loading and Preprocessing**: TensorFlow Image Module Documentation. Used for loading and preprocessing image data (e.g., resizing, decoding JPEG).
- **TensorFlow Datasets**: tf.data API Documentation. Provides input pipelines for optimized data loading.
- **TensorFlow: Prefetch and Parallel Processing**: tf.data.AUTOTUNE Documentation. Optimizes dataset loading and processing using prefetching and parallel mapping.

#### Visualization
- **Seaborn: Cluster and Scatter Plots**: Seaborn Documentation. Used for visualizing clustering results in 2D using PCA components.
- **Matplotlib**: Matplotlib Pyplot Documentation. For creating line plots, scatter plots, and visualizing image batches.    
