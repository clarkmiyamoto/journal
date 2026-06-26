# High Level Note
https://www.math3ma.com/blog/what-is-category-theory-anyway

Information definition of **category**. It is a collection of **objects** related via **morphisms** in a sensible way (e.g. composition & associativity).

Examples

| Category | Objects                      | Morphisms              |
| -------- | ---------------------------- | ---------------------- |
| `Set`    | Sets                         | Functions              |
| `Group`  | Groups                       | Group homomorphisms    |
| `Meas`   | Measurable spaces            | Measurable functions   |
| `VectK`  | Vector spaces over field $K$ | Linear transformations |
By stripping away all the details, you can focus on the objects and relationships between them. 

Relationships between categories are called **functors** 

# Technical Notes
https://arxiv.org/pdf/0905.3010
### Defining a Category
**Definition ((Concrete) Category)** A category $\mathbf C$ consists of
1. A family $|\mathbf C|$ of objects
2. $\forall A, B \in |\mathbf C|$, a set $\mathbf C(A,B)$ of morphisms. This is called the hom-set.
3. $\forall A,B,C \in |\mathbf C|$ and $\forall f \in \mathbf C(A,B)$ and $\forall g \in \mathbf C(B,C)$, a composite $g \circ f \in \mathbf C(A,C)$. The composition operator has the following properties
	1. Associative: $h \circ (g \circ f) = (h \circ g) \circ f$
	2. Units: $\forall A \in |\mathbf C|$ there exists a morphism $1_A \in \mathbf C(A,A)$ called *identity*, s.t. for $f \in \mathbf C(A,B)$ you have $f = f \circ 1_A = 1_B \circ f$.

To show something is a category, you need to make sure the morphisms & composition operator play nicely with each other.

Examples of Concrete Categories

Define $\textbf{Set}$ which is
1. All sets as objects
2. All functions between sets as morphisms.
   That is if $X,Y$ are sets, and $f : X \to Y$ is a function between sets, then $f \in \mathbf{Set}(X,Y)$
3. Ordinary composition of functions.
   For $f : X \to Y$ and $g : Y \to Z$, define $(g \circ f)(x) = g(f(x))$ 
4. Define the identity map: $1_X(x) = x$
It is a concrete category

Define $\mathbf{FdVect}_{\mathbb K}$ 
1. Finite dimensional vector spaces over $\mathbb K$ as objects
2. All linear map between these vector spaces as morphisms
3. Ordinary composition of underlying functions
4. Identity function
It is a concrete category.
*Proof:* Consider a linear map $f(x) = Ax$, these are defined to associative. If the linear map is from $f : \mathbb R^a \to \mathbb R^b$, then your identity is $1_A(x) = \mathbb I_a$ and $1_B(x) = \mathbb I_b$.

Define $\textbf{Comp}$
1. All data types (e.g. Booleans, integers, float) as objects
2. All programs which take data of type $A$ as their input and produces data of type $B$
3. Sequential compositions of programs are defined to be the composition operator. 
4. Program which outputs their input unaltered as identities.

### Properties

*Example* A **monoid** $(M, \cdot, 1)$ is a set together with bindary associative operator $- \cdot - : M \times M \to M$ which admits a unit $1$ (i.e. group w/o inverse).

Equivalently we can define a monoid as a category $\mathbf M$ with a single object $*$. 
- Associative composition operator $$- \circ - : \mathbf M(*,*) \times \mathbf M(*,*) \to \mathbf M(* , *)$$with associative monoid multiplication $\cdot$
- Identity $1_* : * \to *$
- 


**Definition (Isomorphic)** Two objects $A,B \in |\mathbf C|$ are isomorphic if $\exists f \in \mathbf C(A,B)$ and $g \in \mathbf C(B,A)$ s.t. $g \circ f = 1_A$ and $f \circ g = 1_B$. The morphism $f$ is called an isomorphism and we notate $f^{-1} := g$ (and call it the inverse to $f$)

**Definition (Strict Monoidal Category)** is a category for which
1. Objects comes with monoid structure $(|\mathbf C|, \otimes, I)$. That is $\forall A,B,C \in |\mathbf C|$ $$A \otimes (B \otimes C) = (A \otimes B) \otimes C$$ and $$I \otimes A = A = A \otimes I$$
2. For all objects $A,B,C,D \in |\mathbf C|$ there exists an operation $$-\otimes- : \mathbf C(A,B) \times \mathbf C(C,D) \to \mathbf C(A \otimes C, B \otimes D) :: (f,g) \mapsto f \otimes g$$which is associative and has $1_I$ as its unit$$f \otimes (g \otimes h) = (f\otimes g) \otimes h \quad \text{and} \quad 1_I \otimes f = f = f \otimes 1_I$$This operation is called the **monoidal product (or tensor)**.
3. For all morphisms $f,g,h,k$ with matching types $$(g\circ f) \otimes (k \circ h) = (g \otimes k) \circ (f \otimes h)$$(i.e. has tensor product structure in quantum mechanics)
4. For all objects $A,B \in |\mathbf C|$ we have $$1_A \otimes 1_B = 1_{A \otimes B}$$
This structure highly alludes to multi-qubit systems in quantum mechanics. You can think of 
$$
\begin{align}
A \otimes B &:= \text{system } A \text{ and system }  B\\
f \otimes g &:= \text{process } f \text{ and process } g
\end{align}
$$

### Structure Preserving Maps for Categories
Maps which preserve categorical structure are called **functors**. The structure here is *composition* and *identities*.

A functor known in phyiscs is the representation of group. Consider a representation $\rho$ of a group $G$ on vector space $V$; it is a group homomorphism $\rho : G \to \text{GL}(V)$, i.e.
$$
\begin{align}
\rho(g_1 \cdot g_2) = \rho(g_1) \circ \rho(g_2), \quad \text{ and } \quad \rho(1) = 1_V
\end{align}
$$

**Definition (functor)** Let $\mathbf C$ and $\mathbf D$ be categories. A *functor* $$
\begin{align}
F : \mathbf C \to \mathbf D
\end{align}
$$ 
consists of
1. A mapping $$F: |\mathbf C| \to |\mathbf D| :: A \mapsto F(A)$$
2. $\forall A,B \in |\mathbf C|$, a mapping on the morphisms $$F: \mathbf C(A,B) \to \mathbf D(F(A), F(B)) :: f\mapsto F(f)$$
   which preserves identity & composition
	1. For any $f \in \mathbf C(A,B)$ and $g \in \mathbf C(B,C)$ we have $$F(g \circ f) = F(g) \circ F(f)$$
	2. For any $A \in |\mathbf C|$ we have $$F(1_A) = 1_{F(A)}$$
**Definition (Contravariant Functor):** This is a functor $F : \mathbf C \to \mathbf D$ but reverses composition $$
\begin{align}
F(g \circ f) = F(f) \circ F(g)
\end{align}
$$
Orindary functors are referred to as covariant functors.

**Definition (Opposite Category)** The opposite category $\mathbf C^{op}$ of a category $\mathbf C$ is the category with
- Same objects as $\mathbf C$
- Morphisms are reverse, that is $$f \in \mathbf C(A,B) \iff f \in \mathbf C^{op}(B,A)$$To avoid confusion $f \in \mathbf C^{op}(B,A)$ will be denoted as $f^{op}$
- Identities in $\mathbf C^{op}$ are those of $\mathbf C$ and $$f^{op} \circ g^{op} = (g \circ f)^{op}$$
Notice that the $^{op}$ operator is involutive (applying it twice is equivalent to doing nothing).

