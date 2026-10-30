# FCIDump

### struct QfFCIDump

An electronic structure Hamiltonian in FCIDump format.

---

## Definition

The FCIDump format was originally defined by Knowles and Handy, 1989 <sup>[1](#id2)</sup>.
It is a widespread format for exporting electronic structure Hamiltonians in a plain-text file.

The present data structure only stores the information relevant for constructing the second
quantized operator.
However, this implementation goes beyond the original definition by supporting unrestricted
spin data to be loaded. The table below outlines how integrals are associated with spin species
based on the intervals in which the indices fall (assuming a header with `NORB=n`):

| Integral Type      | i          | j          | k          | l          |
|--------------------|------------|------------|------------|------------|
| Constant           | ${0}$      | ${0}$      | ${0}$      | ${0}$      |
| 1-body alpha       | ${0}$      | ${0}$      | $[1,n]$    | $[1,n]$    |
| 1-body beta        | ${0}$      | ${0}$      | $[n+1,2n]$ | $[n+1,2n]$ |
| 2-body alpha-alpha | $[1,n]$    | $[1,n]$    | $[1,n]$    | $[1,n]$    |
| 2-body alpha-beta  | $[1,n]$    | $[1,n]$    | $[n+1,2n]$ | $[n+1,2n]$ |
| 2-body beta-beta   | $[n+1,2n]$ | $[n+1,2n]$ | $[n+1,2n]$ | $[n+1,2n]$ |

The only required values are the 1-body alpha-spin integrals.

The two-body integrals are expected in chemist ordering, $(ij|kl)$, matching the FCIDump
convention.

---

<a id="qf-fcidump-implementation"></a>

## Data layout

The integrals are stored as flattened arrays exploiting their permutational symmetry, which is also
the layout the electronic integral constructors ([Electronic Integrals](qf-electronic-integrals.html.md)) expect. Writing
$n$ for the number of orbitals ([`qf_fcidump_norb()`](#c.qf_fcidump_norb)) and
$\text{npair} = n (n + 1) / 2$, a single data structure contains up to 5 arrays:

| Array         | Length                                | Contents                             |
|---------------|---------------------------------------|--------------------------------------|
| `one_body_a`  | $\text{npair}$                        | The $\alpha$-spin 1-body integrals.  |
| `one_body_b`  | $\text{npair}$                        | The $\beta$-spin 1-body integrals.   |
| `two_body_aa` | $\text{npair} (\text{npair} + 1) / 2$ | The $\alpha\alpha$ 2-body integrals. |
| `two_body_ab` | $\text{npair}^2$                      | The $\alpha\beta$ 2-body integrals.  |
| `two_body_bb` | $\text{npair} (\text{npair} + 1) / 2$ | The $\beta\beta$ 2-body integrals.   |

The 1-body arrays are the flattened lower triangle of an $(n, n)$ matrix, indexed as
`ia = i * (i + 1) / 2 + a` with `a <= i`. The `two_body_aa` and `two_body_bb` arrays are
8-fold (S8) symmetric: the flattened lower triangle of a
$(\text{npair}, \text{npair})$ matrix, whose own two axes are each such a pair index.

#### WARNING
`two_body_ab` is packed *differently* from its siblings. It is only 4-fold (S4) symmetric, so it
holds the **full** $(\text{npair}, \text{npair})$ matrix in row-major order, indexed as
`iajb = ia * npair + jb`. Its row pair indexes the $\alpha$-spin species and its column
pair the $\beta$-spin species, so it is *not* symmetric under exchanging the two pairs.

The $\beta$-spin 1-body and the $\alpha\beta$ / $\beta\beta$ 2-body arrays are only
present for a file carrying unrestricted spin data. They are absent together, so
[`qf_fcidump_is_unrestricted()`](#c.qf_fcidump_is_unrestricted) reports on all three at once and gates the getters that read
them.

The stored values are the integrals as they appear in the file. In particular, they do **not** carry
the conventional factor of $\frac{1}{2}$ that the 2-body operator terms do, so a coefficient of
an operator built via [`qf_ferm_op_from_fcidump()`](#c.qf_ferm_op_from_fcidump) is half the corresponding value returned here.

Building a dense $(n, n)$ or $(n, n, n, n)$ tensor from these arrays (as required by, for
example, sample-based diagonalization tooling) is the caller’s task; the index formulae above are what
that expansion needs.

---
* <a id='id2'>**[1]**</a> 1. 1. Knowles and N. C. Handy, Computer Physics Communications 54 (1989) 75-83.

## Members

### [QfExitCode](qf-exit-code.html.md#c.QfExitCode) qf_fcidump_from_file(char \*file_path, [QfFCIDump](#c.QfFCIDump) \*\*out)

Parses an FCIDump file. 

### Example

Assuming you have an FCIDump file called `molecule.fcidump`, you use this function like so:

```c
QfFCIDump *fcidump = NULL;
QfExitCode exit = qf_fcidump_from_file("molecule.fcidump", &fcidump);

assert(exit == QfExitCode_Success);
```

 * **Parameters:**
  * **file_path** – The path to the FCIDump file. 
  * **out** – A pointer to the pointer that will be set to the parsed FCIDump data structure. It is only written to on success; on failure it is left untouched.
* **Returns:**
  An exit code. This is `[QfExitCode_ValueError](qf-exit-code.html.md#group__qf__exit__code_1gga89daaa15e32524338df290dbd873caf1a869710ce98addf007187616631c85470)` if the file cannot be opened or read, or if it does not honor the FCIDump format (a missing header namelist, a missing `NORB` or `NELEC` field, a malformed `MS2` field, or an unsupported MO energy value).

### void qf_fcidump_free([QfFCIDump](#c.QfFCIDump) \*fcidump)

Frees an existing FCIDump data structure. 

### Example

```c
QfFCIDump *fcidump = NULL;
qf_fcidump_from_file("molecule.fcidump", &fcidump);
qf_fcidump_free(fcidump);
```

 * **Parameters:**
  * **fcidump** – A pointer to the FCIDump data structure to be freed.

### uint32_t qf_fcidump_norb(const [QfFCIDump](#c.QfFCIDump) \*fcidump)

Gets the number of orbitals from an FCIDump. 

This number, $n$, is extracted from the `NORB=n` field in the header of the FCIDump
file.

### Example

```c
QfFCIDump *fcidump = NULL;
qf_fcidump_from_file("molecule.fcidump", &fcidump);
uint32_t norb = qf_fcidump_norb(fcidump);
```

 * **Parameters:**
  * **fcidump** – A pointer to the FCIDump data structure to query.
* **Returns:**
  The number of orbitals.

### uint32_t qf_fcidump_nelec(const [QfFCIDump](#c.QfFCIDump) \*fcidump)

Gets the number of electrons from an FCIDump. 

This number, $n$, is extracted from the `NELEC=n` field in the header of the FCIDump
file.

### Example

```c
QfFCIDump *fcidump = NULL;
qf_fcidump_from_file("molecule.fcidump", &fcidump);
uint32_t nelec = qf_fcidump_nelec(fcidump);
```

 * **Parameters:**
  * **fcidump** – A pointer to the FCIDump data structure to query.
* **Returns:**
  The number of electrons.

### uint32_t qf_fcidump_ms2(const [QfFCIDump](#c.QfFCIDump) \*fcidump)

Gets the spin quantum number from an FCIDump. 

This number, $S$, is extracted from the `MS2=S` field in the header of the FCIDump
file.

### Example

```c
QfFCIDump *fcidump = NULL;
qf_fcidump_from_file("molecule.fcidump", &fcidump);
uint32_t ms2 = qf_fcidump_ms2(fcidump);
```

 * **Parameters:**
  * **fcidump** – A pointer to the FCIDump data structure to query.
* **Returns:**
  The spin quantum number (multiplied by 2 to ensure an integer value).

### bool qf_fcidump_is_unrestricted(const [QfFCIDump](#c.QfFCIDump) \*fcidump)

Checks whether an FCIDump carries unrestricted (spin-dependent) integrals. 

The beta-spin 1-body and the alpha-beta / beta-beta 2-body arrays are present exactly when this
returns `true`; they are absent together. Use it to guard
[`qf_fcidump_get_one_body_tril_b()`](#c.qf_fcidump_get_one_body_tril_b), [`qf_fcidump_get_two_body_tril_ab()`](#c.qf_fcidump_get_two_body_tril_ab) and
[`qf_fcidump_get_two_body_tril_bb()`](#c.qf_fcidump_get_two_body_tril_bb).

### Example

```c
QfFCIDump *fcidump = NULL;
qf_fcidump_from_file("molecule.fcidump", &fcidump);

bool unrestricted = qf_fcidump_is_unrestricted(fcidump);
```

 * **Parameters:**
  * **fcidump** – A pointer to the FCIDump data structure to query.
* **Returns:**
  `true` if the beta-spin integrals are present.

### bool qf_fcidump_has_constant(const [QfFCIDump](#c.QfFCIDump) \*fcidump)

Checks whether an FCIDump provides a constant energy offset. 

Use this to guard [`qf_fcidump_constant()`](#c.qf_fcidump_constant). This is independent of
[`qf_fcidump_is_unrestricted()`](#c.qf_fcidump_is_unrestricted): a spin-restricted file may well carry a constant.

### Example

```c
QfFCIDump *fcidump = NULL;
qf_fcidump_from_file("molecule.fcidump", &fcidump);

bool has_constant = qf_fcidump_has_constant(fcidump);
```

 * **Parameters:**
  * **fcidump** – A pointer to the FCIDump data structure to query.
* **Returns:**
  `true` if the file carried an integral line whose four indices are all zero.

### double qf_fcidump_constant(const [QfFCIDump](#c.QfFCIDump) \*fcidump)

Gets the constant energy offset from an FCIDump. 

This is the value of the integral line whose four indices are all zero, which for an electronic
structure Hamiltonian is the nuclear-repulsion energy.

### Example

```c
QfFCIDump *fcidump = NULL;
qf_fcidump_from_file("molecule.fcidump", &fcidump);

double constant = 0.0;
if (qf_fcidump_has_constant(fcidump)) {
    constant = qf_fcidump_constant(fcidump);
}
```

 #### WARNING
Only call this when [`qf_fcidump_has_constant()`](#c.qf_fcidump_has_constant) returns `true`; calling it otherwise
panics.

* **Parameters:**
  * **fcidump** – A pointer to the FCIDump data structure to query.
* **Returns:**
  The constant energy offset.

### void qf_fcidump_get_one_body_tril_a(const [QfFCIDump](#c.QfFCIDump) \*fcidump, double \*\*out, uint64_t \*out_len)

Provides read-only access to the alpha-spin 1-body integrals. 

#### SEE ALSO
The explanation of the internal data layout, [here](#qf-fcidump-implementation).

### Example

```c
QfFCIDump *fcidump = NULL;
qf_fcidump_from_file("molecule.fcidump", &fcidump);

double *integrals;
uint64_t len;
qf_fcidump_get_one_body_tril_a(fcidump, &integrals, &len);
```

 #### NOTE
This function borrows the FCIDump’s internal buffer rather than copying it. The returned
pointer stays valid only until the data structure is freed; do not free it yourself.

* **Parameters:**
  * **fcidump** – A pointer to the FCIDump data structure whose integrals to access. 
  * **out** – A pointer to the array of doubles into which to write the integrals. 
  * **out_len** – A pointer to the integer into which to write the length of the output array.

### void qf_fcidump_get_one_body_tril_b(const [QfFCIDump](#c.QfFCIDump) \*fcidump, double \*\*out, uint64_t \*out_len)

Provides read-only access to the beta-spin 1-body integrals. 

#### SEE ALSO
The explanation of the internal data layout, [here](#qf-fcidump-implementation).

### Example

```c
QfFCIDump *fcidump = NULL;
qf_fcidump_from_file("molecule.fcidump", &fcidump);

double *integrals;
uint64_t len;
qf_fcidump_get_one_body_tril_b(fcidump, &integrals, &len);
```

#### WARNING
Only call this on a data structure carrying unrestricted spin data. Check
[`qf_fcidump_is_unrestricted()`](#c.qf_fcidump_is_unrestricted) first; calling it otherwise panics.

#### NOTE
This function borrows the FCIDump’s internal buffer rather than copying it. The returned
pointer stays valid only until the data structure is freed; do not free it yourself.

* **Parameters:**
  * **fcidump** – A pointer to the FCIDump data structure whose integrals to access. 
  * **out** – A pointer to the array of doubles into which to write the integrals. 
  * **out_len** – A pointer to the integer into which to write the length of the output array.

### void qf_fcidump_get_two_body_tril_aa(const [QfFCIDump](#c.QfFCIDump) \*fcidump, double \*\*out, uint64_t \*out_len)

Provides read-only access to the alpha-alpha-spin 2-body integrals. 

#### SEE ALSO
The explanation of the internal data layout, [here](#qf-fcidump-implementation).

### Example

```c
QfFCIDump *fcidump = NULL;
qf_fcidump_from_file("molecule.fcidump", &fcidump);

double *integrals;
uint64_t len;
qf_fcidump_get_two_body_tril_aa(fcidump, &integrals, &len);
```

 #### NOTE
This function borrows the FCIDump’s internal buffer rather than copying it. The returned
pointer stays valid only until the data structure is freed; do not free it yourself.

#### NOTE
The values are in chemist ordering, $(ij|kl)$, and carry no factor of
$\frac{1}{2}$.

* **Parameters:**
  * **fcidump** – A pointer to the FCIDump data structure whose integrals to access. 
  * **out** – A pointer to the array of doubles into which to write the integrals. 
  * **out_len** – A pointer to the integer into which to write the length of the output array.

### void qf_fcidump_get_two_body_tril_ab(const [QfFCIDump](#c.QfFCIDump) \*fcidump, double \*\*out, uint64_t \*out_len)

Provides read-only access to the alpha-beta-spin 2-body integrals. 

#### SEE ALSO
The explanation of the internal data layout, [here](#qf-fcidump-implementation).

### Example

```c
QfFCIDump *fcidump = NULL;
qf_fcidump_from_file("molecule.fcidump", &fcidump);

double *integrals;
uint64_t len;
qf_fcidump_get_two_body_tril_ab(fcidump, &integrals, &len);
```

#### WARNING
Only call this on a data structure carrying unrestricted spin data. Check
[`qf_fcidump_is_unrestricted()`](#c.qf_fcidump_is_unrestricted) first; calling it otherwise panics.

#### NOTE
This function borrows the FCIDump’s internal buffer rather than copying it. The returned
pointer stays valid only until the data structure is freed; do not free it yourself.

#### NOTE
The values are in chemist ordering, $(ij|kl)$, and carry no factor of
$\frac{1}{2}$.

#### WARNING
Unlike the other 2-body arrays, this one is only 4-fold (S4) symmetric: it holds the
**full** `(npair, npair)` matrix, whose row pair indexes the alpha-spin and whose
column pair indexes the beta-spin species. It is therefore *not* symmetric under
exchanging the two pairs.

* **Parameters:**
  * **fcidump** – A pointer to the FCIDump data structure whose integrals to access. 
  * **out** – A pointer to the array of doubles into which to write the integrals. 
  * **out_len** – A pointer to the integer into which to write the length of the output array.

### void qf_fcidump_get_two_body_tril_bb(const [QfFCIDump](#c.QfFCIDump) \*fcidump, double \*\*out, uint64_t \*out_len)

Provides read-only access to the beta-beta-spin 2-body integrals. 

#### SEE ALSO
The explanation of the internal data layout, [here](#qf-fcidump-implementation).

### Example

```c
QfFCIDump *fcidump = NULL;
qf_fcidump_from_file("molecule.fcidump", &fcidump);

double *integrals;
uint64_t len;
qf_fcidump_get_two_body_tril_bb(fcidump, &integrals, &len);
```

#### WARNING
Only call this on a data structure carrying unrestricted spin data. Check
[`qf_fcidump_is_unrestricted()`](#c.qf_fcidump_is_unrestricted) first; calling it otherwise panics.

#### NOTE
This function borrows the FCIDump’s internal buffer rather than copying it. The returned
pointer stays valid only until the data structure is freed; do not free it yourself.

#### NOTE
The values are in chemist ordering, $(ij|kl)$, and carry no factor of
$\frac{1}{2}$.

* **Parameters:**
  * **fcidump** – A pointer to the FCIDump data structure whose integrals to access. 
  * **out** – A pointer to the array of doubles into which to write the integrals. 
  * **out_len** – A pointer to the integer into which to write the length of the output array.

---

## Conversion

Operator representations which can be constructed from an instance of
[`QfFCIDump`](#c.QfFCIDump) provide a `qf_*_from_fcidump` function:

### [QfFermionOperator](qf-ferm-op.html.md#c.QfFermionOperator) \*qf_ferm_op_from_fcidump(const [QfFCIDump](#c.QfFCIDump) \*fcidump)

Constructs an :c:struct:`QfFermionOperator` from a :c:struct:`QfFCIDump`. 

### Example

```c
QfFCIDump *fcidump = NULL;
qf_fcidump_from_file("molecule.fcidump", &fcidump);
QfFermionOperator *op = qf_ferm_op_from_fcidump(fcidump);
```

 * **Parameters:**
  * **fcidump** – A pointer to the FCIDump data structure.
* **Returns:**
  A pointer to the created operator.
