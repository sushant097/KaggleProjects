# Cdiscount’s Image Classification

Cdiscount.com generated nearly 3 billion euros last year, making it France’s largest non-food e-commerce company. While the company already sells everything from TVs to trampolines, the list of products is still rapidly growing. By the end of this year, Cdiscount.com will have over 30 million products up for sale. This is up from 10 million products only 2 years ago. Ensuring that so many products are well classified is a challenging task.

In this challenge you will be building a model that automatically classifies the products based on their images. As a quick tour of Cdiscount.com's website can confirm, one product can have one or several images.


### Dataset
The data set Cdiscount.com is making available is unique and characterized by superlative numbers in several ways:

* Almost 9 million products: half of the current catalogue
* More than 15 million images at 180x180 resolution
* More than 5000 categories: yes this is quite an extreme multi-class classification!


Please Note: The train and test files are very large!

* train.bson - (Size: 58.2 GB) Contains a list of 7,069,896 dictionaries, one per product. Each dictionary contains a product id (key: _id), the category id of the product (key: category_id), and between 1-4 images, stored in a list (key: imgs). Each image list contains a single dictionary per image, which uses the format: {'picture': b'…binary string…'}. The binary string corresponds to a binary representation of the image in JPEG format. This kernel provides an example of how to process the data.

* train_example.bson - Contains the first 100 records of train.bson so you can start exploring the data before downloading the entire set.

* test.bson - (Size: 14.5 GB) Contains a list of 1,768,182 products in the same format as train.bson, except there is no category_id included. The objective of the competition is to predict the correct category_id from the picture(s) of each product id (_id). The category_ids that are present in Private Test split are also all present in the Public Test split.

* category_names.csv - Shows the hierarchy of product classification. Each category_id has a corresponding level1, level2, and level3 name, in French. The category_id corresponds to the category tree down to its lowest level. This hierarchical data may be useful, but it is not necessary for building models and making predictions. All the absolutely necessary information is found in train.bson.


**What we learn**
* How to do handle extreme Multi-class classification problem which have more than 5000 categories.
* Data loading, training with PyTorch Lightning.

 