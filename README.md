
## Project Name

Self-Supervised Based Graph Autoencoder for Cancer Gene Identification

![Figure](figure/figure1.png)

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/C0nc/SMG.git
   ```

2. Navigate to the project directory:

   ```shell
   cd SMG
   ```

3. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```

   This will install all the necessary packages specified in the `requirements.txt` file.

## Usage

0. Predefined protein-protein interaction network index:

   ```shell
   ['CPDB', 'IRefIndex', 'PCNet', 'IRefIndex_2015', 'STRINGdb', 'Multinet']
   ```

1. Run to train the model to predict the gene nodes by the semi-supervised transductive learning:

   ```shell
   python main_transductive.py [arguments]
   ```

   Provide the required arguments based on your project's needs. Below are the available arguments:

   - `--ppi`: Choose the train protein-protein interaction network based on the defined network index.
   - `--inductive-ppi`: Choose the test protein-protein interaction network based on the defined network index (when inductive learning is needed).
   - `--expression`: Change the task to essential gene prediction.
   - `--health`: Change the task to health gene prediction.

2. Run to train the model to predict the graph property:

   ```shell
   python main_graph.py --use_cfg [arguments]
   
   - `task`: Choose the architecture GIN_graph/GCN_graph
   ```

3. Run to conduct the post-hoc explanation by the GNNExplainer and the Integrated Gradient:

   ```shell
   python main_transductive.py [arguments] 
  
   - `GE`: Utilize the GraphExplainer to explain the prediction results.
   - `IGE`: Utilize the Integrated Gradient to explain the prediction results.
   ```

## Reference Website

1. You can choose the target gene sets enrichment analysis based on the list in this reference website:

   [https://maayanlab.cloud/Enrichr/](https://maayanlab.cloud/Enrichr/)

2. More information about Cancer Gene can be found at:

   [http://ncg.kcl.ac.uk/](http://ncg.kcl.ac.uk/)

## Data available

Get the data from this Google Drive link:

[https://drive.google.com/file/d/1Kfj2xdCbmRPpRn9s-0BqP33Z5wzoyv2m/view?usp=drive_link](https://drive.google.com/file/d/1Kfj2xdCbmRPpRn9s-0BqP33Z5wzoyv2m/view?usp=drive_link)

## License

This project is licensed under the [MIT License](LICENSE).
