#ifndef PRINTERVIEWDATA_H
#define PRINTERVIEWDATA_H

#include <QString>

/***
 * ViewData struct Rules:
 * -Immutable after creation
 * -No behavior or Qt signals
 * -No pointers to models
 * -Safe to copy
 **/
struct PrinterViewData {
    QString id;
    QString name;
    bool isSystem;
};

#endif // PRINTERVIEWDATA_H
