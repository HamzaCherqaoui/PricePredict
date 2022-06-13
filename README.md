\documentclass[twocolumn,letterpaper,10pt]{article}
\usepackage{times}
\usepackage{helvet}
\usepackage{courier}
\usepackage{txfonts}
\frenchspacing
%\setlength{\pdfpagewidth}{8.5in}
%\setlength{\pdfpageheight}{11in}

%used for pseudocode
\usepackage{algpseudocode}

%used to make charts
\usepackage{pgfplotstable}
\usepackage{pgfplots}

%used for mathematical notation
\usepackage{amsfonts}

%used to control spacing in table captions
\usepackage{subfig}

%used to import images
\usepackage{graphicx}

%used to specify table placement
\usepackage{float}

% Make it so lists don't have extra line spacing:
\usepackage{enumitem}
\setlist{noitemsep}

\usepackage{hyperref} % for \url

% For nice, customizable code listings:
\usepackage{listings}
\lstset{ %http://stackoverflow.com/questions/586572/make-code-in-latex-look-nice
	language=Java,
	basicstyle=\footnotesize,       % font size
	numbers=left,                      % where line numbers appear
	numberstyle=\footnotesize,   % size of line numbers
	stepnumber=1,                    % how often line numbers appear
	numbersep=5pt,                   % space between line numbers and code
	showstringspaces=false,       % whether or not to show a _ in strings
	frame=single,                       % what kind of frame should be around the code
	xleftmargin=1.5em,
	xrightmargin=1em,
	framexleftmargin=0em,
	%rulecolor=\color{black},         % for the frame
	tabsize=2,                             % set the number of spaces for tabs
	breaklines=true,                     % set whether or not linebreaking should occur
	breakatwhitespace=true,          % set whether or not linebreaking should occur only at spaces
	lineskip={-1.0pt}
	%commentstyle=\color{mygreen}, % color for code comments
	%numberstyle=\color{mygray}, % style for line numbers
}


\renewcommand{\today}{}

%\usepackage{plain} % for the bibliography

\title{Housing Price Prediction Models}

\author{Hamza Cherqaoui, Ihab El Kerdoudi \\
\\
DePauw University\\
Greencastle, IN 46135, U.S.A.\\
}

%define mathematical equations for repeated use
\newcommand{\sigmoid}{$$S(t)=\frac{1}{1+e^{-t}}$$}
\newcommand{\sigmoidprime}{$$S'(t)=S(t) \cdot (1 - S(t))$$}

\begin{document}

\maketitle

\begin{abstract}

Estimating cost is the foundation for any business and it proves extremely critical for planning a job's schedule and budget. This paper explores predictions on housing prices based on a given data set , it delves into the process of training models and generating predictions using linear regression and other types of regressions. This project also serves the purpose of testing pre-processing, algorithms, and parameterization in order to find the best housing price estimator. The data set used in this research provides information on houses in a suburban area, and through the course of this research it under goes various attribute adjustments and processing and is fitted to various models that generate predictions on prices with moderately high accuracy.

\end{abstract}



\section{Data Description}
\label{sec:NN}

{\it The data set used in this research was provided by a Kaggle competition by the title of  "House Prices - Advanced Regression Techniques", it is called The Ames Housing dataset and it was compiled by Dean De Cock for the purpose of education in data science. It's a good alternative to the Boston Housing dataset as it is more modernized and expanded. The houses in this data set each posess 81 attributes ranging from the property's sale price in dollars, its lot size in square feet all the way to the type of roofing used in that property. Amongst these 81 attributes, 38 of them were numerical attributes (i.e: 'OverallQual': Overall material and finish quality) , and  43 categorical attributes such as the height of the basement which are ordered from "Excellent" to "No Basement". This data set contains certain attributes with missing values, especially the numerical attributes, it also contains some ordinal attributes that have values ranging from "Ex" which stands for "Excellent" to "Po" which stands for "Poor" such as the "ExterQual" attribute which evaluates the quality of the material on the exterior of the property. This data set also contains attributes for the location of each house which need to undergo moderate pre-processing in order to be later fitted to our models. 
{\it  The nature of these attributes and a good understanding of how to process their values is a very critical step in order to utilize the integrity of the data set and increase the accuracy of our models.}

{\it Table~\ref{tab:Table 1} shows the attributes used in this research which were classified into broader classes. The table also shows the type of values they contain. }


\begin{table}[h!]
  \begin{center}
    \caption{Table of Attributes used in this research}
    \label{tab:Table 1}
    \begin{tabular}{l|c|r} % <-- Alignments: 1st column left, 2nd middle and 3rd right, with vertical lines in between
      \textbf{Attribute Class} & \textbf{Example} & \textbf{Type}\\
      \hline
      sqfAtts &'1stFlrSF' & Numerical\\
      yearAtts &YearBuilt' & Numerical\\
      qualitativeAtts &'OverallQual' & Ordinal\\
      locationAtts &'Street' & Nominal\\
      conditionAtts &'Condition1' & Nominal\\
      ExtAtts&'Exterior1st' & Nominal\\
      roomAtts&'TotRmsAbvGrd' & Nominal\\
      RoofAtts&'RoofStyle' & Nominal\\
    \end{tabular}
  \end{center}
\end{table}

{\it Working with this data set has required the use of many pre-processing techniques and helped make the data more adequate to our model when fitting and predicting.}

\section{Experiment}
\subsection{Pre-Processing}

{\it As mentioned in the previous section, the data set at hand comprised many attributes with missing values, thus the first process that was implemented in this project is to handle this case of missing values especially in numerical attributes. We started this step of the pre-processing by retrieving all the attributes that have numeric values first using the available helper function getAttrsWithMissingValues(), then out of these attributes we recovered those that contained missing values with another helper function. In order to handle this issue, missing values were replaced with the mean for each attribute! }

{\it The next step in this process was to add some new attributes to our predictors list as part of our feature-engineering efforts. Using some of the square footage attributes we first created a new attribute named 'PropertySF' which adds both the square footage of the first floor and the Garage Living area. We then created an age attribute for each property by subtracting the 'YearBuilt' attribute by the current year, this new attribute was named 'YearsOld'. Another trivial attribute was the total square footage of the property which adds values of square footage from the garage area, the total basement area, the first and second floor footage as well as the garage living area! }

{\it After adding new attributes to our predictors, we implemented functions to handle conversions from non-numeric values to numeric. As previously mentioned, attributes with ordinal values were converted to numerical by mapping numbers in ascending order to each corresponding value of such attribute. For the attributes 'Condition 1' and 'Condition 2', we used a helper function that uses a dictionary to map certain values to their respective numeric representations and we followed the same procedure for Exterior attributes and Location attributes. Figure~\ref{fig:figure1} shows a snippet of the code used in this process for the qualitative attributes.}

\begin{figure}[t]
	\begin{lstlisting}
def ordToNumForQual(trainDF,testDF,col):	
    
    trainDF[col] = trainDF[col].map(lambda v:  4 if v=="Ex" else v)
    trainDF[col] = trainDF[col].map(lambda v:  3 if v=="Gd" else v)
    trainDF[col] = trainDF[col].map(lambda v:  2 if v=="TA" else v)
    trainDF[col] = trainDF[col].map(lambda v:  1 if v=="Fa" else v)
    trainDF[col] = trainDF[col].map(lambda v:  0 if v=="Po" else v)

}
	\end{lstlisting}
	\caption{This method, maps numerical values to each value of the 'qualitativeAtts'}
	\label{fig:figure1}
\end{figure}

{\it After converting our values to numerical values, we still needed to fill in missing values for this converted attributes, so we wrote a function to map the mode of the attribute for each missing value. In this case, we used this helper function for both location attributes and Exterior attributes.}

{\it Following this conversion from categorical to numerical values, we did one hot encoding for the ordinal attributes we converted and used the built-in getdummies function in pandas. This part of the pre-processing was especially intricate as some attributes when one-hot encoded and included in our predictors, significantly reduced our average cross-validation score or didn't improve it at all, so we made sure to select only the attributes that improved said average.}


\subsection{Algorithms and Hyperparameterization}


{\it This research used numerous estimators and built off of a Linear Regression model to fit the data and compute an avergae score in the early stages. In order to come up with the best possible accuracy, five more experiments were implemented where various models were built and fit to the data set, we used these models in this respective order: an Elastic Net Regressor, a Bayesian Ridge Regressor, a Gradient Boosting Regressor, a Ridge Regressor and a Lasso Regressor. All of these models yielded different results at the end. }

{\it Elastic net is a type of regularized linear regression and it combines two popular penalties, specifically the L1 and L2 penalty functions.\cite{source}. The objective function for this type of Regressor is:}

\[\\min _{w} \frac{1}{2 n_{\text {samples }}}\|X w-y\|_{2}^{2}+\alpha \rho\|w\|_{1}+\frac{\alpha(1-\rho)}{2}\|w\|_{2}^{2}\] 

{\it Bayesian Ridge Regression is more robust to ill-posed problems, and we used it to generate an average CV score to compare to other models like Lasso which is "a linear model that estimates sparse coefficients", it is used over regression methods for a more accurate prediction by using shrinkage.\cite{sklearn}.Figure \ref{fig:graph}, shows Lasso and Elastic Net Paths!\cite{sklearn}}

\begin{figure}
  \includegraphics[width=\linewidth]{graph.png}
  \caption{Lasso and Elastic-Net Paths}
  \label{fig:graph}
\end{figure}

{\it We have also used a Linear Ridge regressor which yielded results close to those of the linear regression model. The most significant model that was used in this research is the Gradient Boosting Regressor, it yielded the highest CV score average because it uses an ensemble algorithm that produces multiple groups of models to converge to one best fitting model.}

{\it Our attempt at tuning our hyper parameters for the gradient boosting regression model didn't yield increased results, we believe partly due to a misimplementation when using hyper parameters like the learning rate and n estimators for the model. The "tuned" gradient boosting regressor reported a CV average score lower than that of the regular one. }

\subsection{Results}

{\it After pre-processing the data and building various models, the resulting predictions were very interesting and had a lot to say about the tuning of the models and how our pre-processing worked. On one hand the cross validation score obtained from our linear regression model was about 0.807 approximately, other models like Lasso, Ridge, and Elastic-Net had similar results that vary a little. The Bayesian Ridge slightly out performed these models and yielded a score of 0.810. However, the Gradient Boosting Regression model had the best results out of all the other models used in this research having an average cross validation score of 0.898.}
{\it Table \ref{tab:Table 2}, shows the average cross validation scores for all six models we used in descending order.}

\begin{table}[h!]
  \begin{center}
    \caption{Table of Attributes used in this research}
    \label{tab:Table 2}
    \begin{tabular}{l|c} 
      \textbf{Model} & \textbf{CV Average Score} \\
      \hline
      Gradient Boosting  & 0.8982\\
      Bayesian Ridge & 0.8108\\
      Ridge & 0.8075 \\
      Linear Regressor & 0.8066  \\
      Lasso & 0.8066 \\
      Elastic-Net & 0.8003 \\
    \end{tabular}
  \end{center}
\end{table}


\subsection{Analysis}

{\it The results of this research vary greatly and give us an insight on how different models perform using the same data set. While other models like the linear regression one or the Lasso model had decent average scores, other models like the Gradient Boosting regressor had impressive results. To better illustrate this difference, let's compare how Lasso and Elastic-Net operate and compare them with how our highest scoring model woks. Lasso is usually used for simple, sparse models with fewer parameters which is the opposite in our case with the The Ames Housing data set. Elastic-Net is used for the same purpose, however it adds one more level of intricacy by adding regularization penalties to the loss function during training. In addition, Elastic-net is useful when there are multiple attributes that are correlated with one another, in this case, Lasso is more likely to pick one of these attributes at random, while elastic-net is likely to pick both.\cite{sklearn}}

{\it In comparison to the other models, gradient boosting has a comparative advantage because boosting is a strong process that utilizes many models to improve earlier models' mistakes. This is the main reason why the Gradient Boosting model greatly outperformed the rest of the models, because it relies on "the intuition that the best possible next model, when combined with previous models, minimizes the overall prediction error." \cite{article}}

\section{Conclusion}

{\it The results of this project showed that data pre-processing is one of the main factors that produce an accurate model. When we used only numerical predictors at first, namely the square footage attributes, the model was not very accurate and was falling short of a 0.5 CV average score. But, processing categorical attributes like roofing attributes, and qualitative attributes improved the accuracy of our models and helped us achieve greater results especially with the gradient boosting regression model. In conclusion, in order to produce a high accuracy model, each data attribute has to be closely scrutinized and processed in the right way, whether it be converted to a different type or one-hot encoded, paired with an adequate high scoring model for the data set. Hyper parameter optimization is also an excellent way to improve on a model's accuracy.  }

\bibliographystyle{plain}
\bibliography{FinalPaper}

\end{document}
