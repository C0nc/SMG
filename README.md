

## Project Name

Self-Supervised based Graph Autoencoder for Cancer Gene Identification

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

1. Run to train the model to predict the gene nodes by the semi-supervised transductive learning:

   ```shell
   python main_transductive.py [arguments]
   ```

   Provide the required arguments based on your project's needs. Below are the available arguments:

   - `--ppi`: Choice the train protein-protein Intecation network based on defined network index.
   - `--inductive-ppi`: Choice the test protein-protein Intecation network based on defined network index (Whne inductive needs).
   - `--expression`: Change the task into the essential gene prediction.
   - `--healthL`: Change the task into the health gene prediction.

 2. Run to train the model to predict the graph property.:

   ```shell
   python main_graph.py [arguments]
   ```

   Provide the required arguments based on your project's needs. Below are the available arguments:

   - `--ppi`: Choice the train protein-protein Intecation network based on defined network index.
   - `--inductive-ppi`: Choice the test protein-protein Intecation network based on defined network index (Whne inductive needs).
   - `--expression`: Change the task into the essential gene prediction.
   - `--healthL`: Change the task into the health gene prediction.

 3. Run to conduct the post-hoc explaination by the GNNExpainer and the Ingrated Gradient:

   ```shell
   python explain.py [arguments]
   ```

4. Run to conduct the Gene Set Enrichment Analyse based on the Ingarated Gradient Results, you can choice your target background gene set:

   ```shell
   python enrich.py [arguments]
   ```

## Data available:

Get the needed data from this googel driver link:



## License

This project is licensed under the [MIT License](LICENSE).

