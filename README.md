# Personalized Recommendation System For Instacart

## Situation
In the highly competitive online grocery delivery sector, businesses like Instacart strive to understand customer behaviors, preferences, and segmentation to enhance service personalization and operational efficiency. The e-commerce landscape demands innovative solutions to improve customer experience and drive sales growth[1].

## Task
We aimed to dissect the extensive Instacart dataset available on Kaggle to uncover actionable insights on customer purchasing behaviors. Our goal was to develop a comprehensive analysis that would lead to the creation of a personalized recommendation system[1].

## Action
Our approach encompassed several key steps:

1. **Data Exploration**: We conducted an in-depth analysis of customer purchase behavior, revealing a 59% reorder rate and an average of 20 items per order, with a preference for food and personal care products[1].

2. **Customer Segmentation**: Utilizing K-means clustering and PCA for dimension reduction, we segmented customers into groups based on their aisle preferences. This analysis revealed four distinct clusters, with three primarily focused on fresh fruits/vegetables and one on baby food formula[1].

3. **Market Basket Analysis**: We employed the Apriori algorithm for association rule mining to identify patterns in product purchases and discover associations between different items[1].

4. **Recommendation System Development**: We created a hybrid recommendation system that combines collaborative filtering and content-based approaches to provide personalized product suggestions[1].

5. **Dashboard Creation**: We developed an interactive dashboard to visualize insights and facilitate data-driven decision-making[1].

## Result
The recommendation system and dashboard facilitated data-driven decisions, setting the stage for a personalized marketing strategy to enhance customer engagement and boost sales. Key outcomes include:

- **Enhanced Customer Segmentation**: Improved understanding of customer groups and their preferences.
- **Personalized Shopping Experience**: Tailored product recommendations based on individual and group purchasing patterns.
- **Efficient Marketing and Promotions**: Data-driven insights to guide targeted marketing efforts[1].

## Dataset
The analysis was based on the Instacart dataset, which includes:

- **Products**: Information about each product, including its name, associated aisle and department.
- **Order Product Set**: Products in each order and whether the product was reordered.
- **Department**: Each department's ID and name.
- **Aisle**: Each aisle's ID and name.
- **Order**: Information about each order, associated User ID and order time[1].

## Methodology Highlights

### Data Preparation
We aggregated purchase data by customer and aisle using cross-tabulation, creating a comprehensive view of customer preferences across different product categories[1].

### Customer Segmentation
- Performed PCA for dimension reduction, selecting 78 components to explain at least 75% of the variance.
- Utilized k-means clustering with k=4, determined through Elbow Method and Silhouette Score analysis[1].

### Association Rule Mining
Implemented the Apriori algorithm with a minimum support threshold of 0.01 and a lift of 1 to identify significant product associations[1].

### Recommendation System
Developed a hybrid system combining collaborative filtering and content-based approaches to provide personalized product suggestions and maximize cross-selling opportunities[1].

## Dashboard Demo

Our interactive dashboard showcases the insights gained from this analysis and demonstrates the functionality of our recommendation system. It's designed to engage visitors from the moment they land on the homepage and maximize opportunities for additional sales.

[Access the Dashboard Demo](https://personalized-recommendation-system-for-instacart-wisdom.streamlit.app/)

## Business Value
The project delivers significant business value through:
- Improved customer understanding and segmentation
- Enhanced personalization of the shopping experience
- More efficient and targeted marketing strategies
- Potential for increased sales through cross-selling and upselling[1]
