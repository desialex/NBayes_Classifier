# Naive Bayes implementation
__(with Bernoulli features and tf-idf weighting)__

Given a training set of documents (each labelled with a class), and a set of <img src="https://render.githubusercontent.com/render/math?math=K"> classes, we estimate a NB text classification method with Bernoulli features as follows:

1. We define the vocabulary as the set <img src="https://render.githubusercontent.com/render/math?math=V"> where the number of types in the vocabulary defines the dimension of the feature vectors.

2. We define the document likelihoods in each class <img src="https://render.githubusercontent.com/render/math?math=P(\mathcal{D}|C_k)"> in terms of the individual word likelihoods <img src="https://render.githubusercontent.com/render/math?math=P(\textbf{w}|C_k)"> (under an assumption of independant occurrence):  
<img src="https://render.githubusercontent.com/render/math?math=P(\mathcal{D}|C_k) = P(\textbf{w}|C_k) = \prod_{t=1}^{|V|}[w_tP(w_t|C_k)+(1-w_t)(1-P(w_t|C_k))]"> and we take the <img src="https://render.githubusercontent.com/render/math?math=\log"> of that likelihood in order to convert the product to a sum.

3. We estimate the word likelihoods as: <img src="https://render.githubusercontent.com/render/math?math=\hat{P}(w_t|C_k)=\frac{n_k(w_t)}{N_k}">, where <img src="https://render.githubusercontent.com/render/math?math=n_k(w_t)"> is the number of documents of class <img src="https://render.githubusercontent.com/render/math?math=k"> containing word <img src="https://render.githubusercontent.com/render/math?math=w_t"> for <img src="https://render.githubusercontent.com/render/math?math=k=1, ..., K, t=1, ..., |V|"> and where <img src="https://render.githubusercontent.com/render/math?math=N_k"> is the number of documents labelled with class <img src="https://render.githubusercontent.com/render/math?math=k">, for <img src="https://render.githubusercontent.com/render/math?math=k=1, ..., K">

4. We estimate the class priors as: <img src="https://render.githubusercontent.com/render/math?math=\hat{P}(C_k)=\frac{N_k}{N}">

5. To classify an unlabelled document <img src="https://render.githubusercontent.com/render/math?math=\mathcal{D}">, we estimate the posterior probability for each class <img src="https://render.githubusercontent.com/render/math?math=k"> as:
<img src="https://render.githubusercontent.com/render/math?math=P(C_k|\textbf{w}) \propto P(\textbf{w}|C_k)P(C_k) \propto P(C_k)\prod_{t=1}^{|V|}[w_tP(w_t|C_k)+(1-w_t)(1-P(w_t|C_k))]"> of which we again take the <img src="https://render.githubusercontent.com/render/math?math=\log"> to turn the multiplication between prior and likelihood into a sum.

6. We select the label as the class with highest probability.

We implemented our own __tf-idf__ weighting method (with only Python, Numpy and Pandas) using a normalized log term frequency adjusted for document length, where <img src="https://render.githubusercontent.com/render/math?math=N"> is the total number of documents in the corpus <img src="https://render.githubusercontent.com/render/math?math=D"> and <img src="https://render.githubusercontent.com/render/math?math=n_t"> is the number of documents in which the term <img src="https://render.githubusercontent.com/render/math?math=t"> appears.
<img src="https://render.githubusercontent.com/render/math?math=\text{tf-idf}(t,d,D) = \log\left(1+\frac{f_{t,d}}{\sum_{t' \in D}f_{t',d}}\right) \log\left(\frac{N}{n_t}\right)">
