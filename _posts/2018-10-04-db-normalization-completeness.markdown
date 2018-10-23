---
layout: post
title:  "Optimal Hierarchical Indexing Is NP-Complete"
---
{% include math.html %}

## Introduction
I came across something interesting in the footnotes on the Dynamic Programming section of Steven Skiena's Algorithms Design Manual. He mentioned a probem involving a Prolog compiler optimization in which the compiler was to match arguments of functions to definitions of functional behavior defined in the source code. I produced an example below.

$$p(a, a) = f(a)$$
$$p(a, b) = f(a) * g(b)$$
$$p(b, a) = f(a) + g(b)$$
$$p(b, b) = h(b)$$

, where $$a$$ and $$b$$ are constants defined elsewhere in the code. The Prolog folks wanted to optimize their compiler to find rules given arguments or forms of arguments. For example, using placeholder variables $$X$$ and $$Y$$, $$p(X, X)$$ would match the first and last rules while $$p(X,Y)$$ would match any of them. However, they had an important constraint that made the problem relavent to the dynamic programming section. They wanted to store their rule sets in a way that respected the order in which they were defined. Let's go back to the above definitions, this time rewriting the rules with their line numbers as if this represented Prolog code and the function definition with its defining arguments.

|arguements|line number|
|---|---|
|aa|1|
|ab|2|
|ba|3|
|bb|4|

Hmm... this is starting to look like a key value store... As Skiena describes the problem, the Prolog folks wanted to minimize the total number of edges in the graph representation of their tree. Skiena describes this being for computational efficiency, Though the reason for this is still opaque to me. However, if we imagine this tree is using pointers for each branch, we end up optimizing the size of the data structure if we minimize edges. Hence there's a more general application here. In fact, if we leave leave out the part about respecting the order in which the definitions appeared in the source code, we end up with a problem of the form "encode a set of strings with the smallest tree structure possible given each node will represent a character". This is very interesting in its own right, and Steven Skiena adressed this more general problem in a 1997 paper entitled, [Trie-Based Data Structures for Sequence Assembly](https://pdfs.semanticscholar.org/73a8/ad0364bfe640abd650a23cc3b9c14ea7415b.pdf) with applications to efficiently storing and searching DNA sequences. He ultimately proved the problem NP-Complete.

## Application to Database Optimization

Take a good hard look at figure Figure 1 from Trie-Based Data Structures for Sequence Assembly. It's the same figure in The Algorithm Design Manual. The figure represents two ways of looking up a key stored as a sequence of characters. You could organize your collection of strings with a tree that branches at indexes 1, 2, 3 (in that order) and end up with 12 eduges. Alternatively, you only get 8 edges if you branch at indexes 2, 3, 1. Each edge would correspond to a pointer within our tree structure. Even though each tree results in three pointer dereferences per lookup, there are more pointers within the tree structure depicted on the left than on the right. Hence the tree structure on the right is more space efficient than the tree on the left. Minimizing edges will minimize the memory footprint of our data structure.

Now suppose we stored each of these strings and any data associated with them in a relational database. We'll start with one big table and each character gets its own column. We might also imagine one or more "value" columns that store some data associated with the string composed of the other "key" (character) columns. Our goal is to create a hierarchy of indexes on the characters that minimizes the storage (or memory of if you're using an in-memory database) footprint of the data. We would do this by breaking up the long form representation of characters into small

|character 1|character 2|character 3|line number/data|
|---|---|---|
|a|a|a|1|
|b|a|a|2|
|c|b|b|3|
|d|b|b|4|

Let's imagine we created an index on the character columns to map to the data column. As we know, this will end up sorting the table according to one character column, then defaulting to the next one, and so on. How might we go about deciding on optimal precidence to sort our columns? 

To analyze this systematically, recall databases traditionally construct B-trees on indexed columns. The B-tree takes up $$O(N)$$ space, and searches take worst case $$\log(N)$$ queries (in practice these are likely disk reads or cache misses if the structure is stored in memory). Ultimately, we're going to be searching one giant B-tree (or more likely a B+-tree if we're mantaining some ordering with respect to the data). However, each column is searched according to the order of precidence assigned within the index and *narrows down* the number of elements to search in the next columns. Traditionally, the adage is to index on columns with higher cardinality first, and lowest cardinarlity last. In general, the amortized lookup time is going to be something like:

$$O\left(\sum_{x_1,\dots,x_n}\log\left(N_{x_1}\right)\log\left(N_{x_2}|x_1\right)\right)\dots\log\left(N_{x_n}\vert x_{n-1},\dots,x_1\right)\right)$$

, where $$x_i$$ is a character value of the $$i$$th column, $$N_{x_i}$$ is the number of rows with $$x_i$$ in column $$i$$, and vertical lines denote conditioning on other columns (as in probability). This is more illustrative when cast in terms of conditional probabilities using

$$N_{x_i}\vert x_{i-1},\dots,x_{1} = N_i p\left(x_i\vert x_{i-1},\dots,x_1\right) = N_i p\left(x_i,\dots,x_1\right) / p\left(x_{i-1},\dots,x_1\right) $$

, where $$N_i$$ is the cardinality of column $$i$$. Hence the complexity can be cast as

$$O\left(\sum_{x_1,\dots,x_n}\prod_{i=1}^n \log\left(N_{i}\p\left(x_i\vert x_{i-1},\dots,x_{1}\right)\right)\right)\right)$$

<table 1 with one column and foreign key> <table 1 with foreign key and rest of columns>

## Hash Function Domains As a Special Case
