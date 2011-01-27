#ifndef __ASSERT_H
#define __ASSERT_H

//comment this line out to turn off asserts globally
#define ASSERTS_ON


#ifdef ASSERTS_ON
#define ERR_BUFFER_SIZE 2048
#define PY_ASSERT(x,message)                                            \
    if (!(x)) {                                                         \
        char msgbuf[ERR_BUFFER_SIZE];                                  \
        snprintf(msgbuf, ERR_BUFFER_SIZE, "%s in %s:%d", (message), __FILE__, __LINE__); \
        PyErr_SetString(PyExc_AssertionError, msgbuf);                  \
        return NULL;                                                    \
    }
#else //ifdef ASSERTS_ON
#define PY_ASSERT(x,message)
#endif


#endif
