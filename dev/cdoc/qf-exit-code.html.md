# QfExitCode

### enum QfExitCode

Integer exit codes returned to C. 

*Values:*

### enumerator QfExitCode_Success

`0`: Success. 

### enumerator QfExitCode_CInputError

`100`: Error related to data input. 

### enumerator QfExitCode_NullPointerError

`101`: Unexpected null pointer. 

### enumerator QfExitCode_AlignmentError

`102`: Pointer is not aligned to expected data. 

### enumerator QfExitCode_IndexError

`103`: Index out of bounds. 

### enumerator QfExitCode_DuplicateIndexError

`104`: Duplicate index. 

### enumerator QfExitCode_ValueError

`200`: Error related to some in- or output value. Refer to the function documentation for more details.
