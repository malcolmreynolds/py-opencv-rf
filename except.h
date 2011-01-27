#ifndef __EXCEPT_H
#define __EXCEPT_H

#define ERR_MSG_BUF_SIZE 2048
#define SET_ERROR_STRING(error_object, message)       \
    {  \
        char msgbuf[ERR_MSG_BUF_SIZE]; \
        snprintf(msgbuf, ERR_MSG_BUF_SIZE, "%s:%d %s", __FILE__, __LINE__, (message)); \
        PyErr_SetString((error_object), msgbuf); \
    } \

#endif
