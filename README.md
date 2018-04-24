# data-science-tools

|                                                                                                                             | 
|-----------------------------------------------------------------------------------------------------------------------------| 
| Libraries,Category,Pithy Statement About It,Good for...,Strengths,Weaknesses                                                | 
| PyTorch,ML,A replacement for NumPy to use the power of GPUs,"- Speeding up workloads by utilizing GPUs (Tensors and entire  | 
| neural networks can                                                                                                         | 
|  be moved onto GPU using the .cuda method)                                                                                  | 
| - can utilize n GPUs with nn.DataParallel                                                                                   | 
| - deep learning research that requires                                                                                      | 
|  maximum flexibility and speed                                                                                              | 
| - love working with Pythonic libraries                                                                                      | 
| - doing research","- Interoperable with NumPy                                                                               | 
| - Flexibility to implement highly                                                                                           | 
| customized NNs                                                                                                              | 
| - Can utilize GPUs                                                                                                          | 
| - supports dynamic length inputs                                                                                            | 
| - debug with standard Python                                                                                                | 
| debugging tools (pdb, etc.)                                                                                                 | 
| - backed by Facebook","- In beta                                                                                            | 
| - must expose model through a Flask API or                                                                                  | 
| something like that"                                                                                                        | 
| Scikit-learn,ML,,,,                                                                                                         | 
| Keras,ML,,,,                                                                                                                | 
| TensorFlow,ML,Google's ML Framework,"- when you need to deploy a model to a mobile platform                                 | 
| - when you need large scale distributed training ability                                                                    | 
| - want a big community and other online learning resources                                                                  | 
| like MOOCs","- Tensorboard web app is                                                                                       | 
| built in to allow viewing model                                                                                             | 
| performance in a nice UI                                                                                                    | 
| - Supports distributed training                                                                                             | 
| - easy to deploy model to                                                                                                   | 
| specialized gRPC server                                                                                                     | 
| - can deploy to mobile                                                                                                      | 
| - Used extensively in production                                                                                            | 
| systems at major companies                                                                                                  | 
| like Google","- When you write in TensorFlow sometimes you feel                                                             | 
| that your model is behind a brick wall with several                                                                         | 
| tiny holes to communicate over                                                                                              | 
| - limited support for dynamic length inputs                                                                                 | 
| - special debugger (tfdbg) that must be used                                                                                | 
| separately from pdb (allows to evaluate tensorflow                                                                          | 
| expressions at runtime and browse all tensors                                                                               | 
| and operations in session scope)"                                                                                           | 
| CNTK,ML,Microsoft's ML Framework,,"- 2 to 4 times faster than                                                               | 
| TensorFlow for LSTM ",                                                                                                      | 
| Keras,ML,focus on enabling fast experimentation,,"- works with Tensorflow,                                                  | 
| CNTK, or Theano as backend                                                                                                  | 
| - it’s easy to compare performance                                                                                          | 
| of various neural network tasks                                                                                             | 
| using the same Keras code just                                                                                              | 
| by switching its backend between                                                                                            | 
| CNTK and TensorFlow.",                                                                                                      | 
| Theano,ML,retired lib,,,                                                                                                    | 
| MXNet,ML,,,,                                                                                                                | 
| XGBoost,ML,,,,                                                                                                              | 
| Caffe2,,,,,                                                                                                                 | 
| ,,,,,                                                                                                                       | 
| NumPy,"Data                                                                                                                 | 
| Preprocessing",,,,                                                                                                          | 
| Pandas,"Data                                                                                                                | 
| Preprocessing",,,,                                                                                                          | 
| ,,,,,                                                                                                                       | 
| matplotlib,data viz,,,,                                                                                                     | 
| seaborn,data viz,,,,                                                                                                        | 
| bokeh,data viz,Interactive data viz,Presenting interactive charts in web browsers,,                                         | 
| plotly,data viz,,,,                                                                                                         | 
| tensorboard,data viz,Tensorflow model performance visualization tool,,,                                                     | 
| ,,,,,                                                                                                                       | 
| Pillow,Data loading,Useful for working with image data,,,                                                                   | 
| Open CV,Data loading,Useful for working with image data,,,                                                                  | 
| torchvision,Data loading,Useful for working with image data,,,                                                              | 
| scipy,Data loading,Useful for working with audio data,,,                                                                    | 
| librosa,Data loading,Useful for working with audio data,,,                                                                  | 
| Raw Python,Data loading,Useful for working with textual data,,,                                                             | 
| NLTK,Data loading,"Intended to facilitate teaching and                                                                      | 
| research of NLP and the related fields","- the data scientist who wants to fine-tune the model                              | 
| because it has nine different stemming algorithms                                                                           | 
| - working with multi-lingual data because different                                                                         | 
| algorithms tend to perform better on different                                                                              | 
| languages                                                                                                                   | 
| - working with strings as the fundamental inputs and                                                                        | 
| outputs of library calls (rather than objects)",,                                                                           | 
| SpaCy,Data loading,"- Ruby on Rails of Natural Language Processing                                                          | 
| - large-scale information extraction tasks. If your application                                                             | 
| needs to process entire web dumps, spaCy is the library                                                                     | 
|  you want to be using                                                                                                       | 
| - spaCy is the best way to prepare text for deep learning","- handling textual data                                         | 
| - carefully memory-managed Cython                                                                                           | 
| - fastest in world (according to independent research)                                                                      | 
| - interoperates seamlessly with                                                                                             | 
| TensorFlow, PyTorch, scikit-learn, Gensim                                                                                   | 
| - the developer who just wants a stemmer to use, because it                                                                 | 
| only has 1                                                                                                                  | 
| - working with textual data in an OO way because it's API                                                                   | 
| returns objects rather than strings                                                                                         | 
| - super fast POS tagging compared to NLTK",,-                                                                               | 
| ,,,,,                                                                                                                       | 
| ,,,,,                                                                                                                       | 
| ,,,,,                                                                                                                       | 
| CuBLAS,HPC,,,,                                                                                                              | 
| MKL,HPC,,,,                                                                                                                 | 
| CuDNN,HPC,,,,                                                                                                               | 
