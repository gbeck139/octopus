#ifndef PROFILEMANAGER_H
#define PROFILEMANAGER_H

#include <QObject>
#include <QMap>
#include <QList>

#include "printerprofile.h"
#include "printerviewdata.h"
//#include "materialprofile.h"
//#include "processprofile.h"

class ProfileManager : public QObject
{
    Q_OBJECT
public:
    explicit ProfileManager(QObject *parent = nullptr);

    // View
    QList<PrinterViewData> getSystemPrintersForView() const;  //////////////
    QList<PrinterViewData> getUserPrintersForView() const; //////////////

    QString getActivePrinter(); //////////////
    PrinterViewData getActivePrinterDataForView(); //////////////

    // Methods
    void setActivePrinter(const QString& printerId); //////////////

    void addUserPrinter(const PrinterProfile& profile); //////////////
    void updateUserPrinter(const PrinterProfile& profile); //////////////
    void deleteUserPrinter(const QString& printerId); //////// // Delete User Printer << currently used for debugging purposes

    // void savePrinterProfile(const PrinterProfile* profile); ///// DELETED

signals:
    void printersChanged(); // list structure changes
    void activePrinterChanged(const QString& printerId); // selection changed
    void activePrinterDataChanged(const QString& printerId); // contents changed

private:
    // Helper methods
    void loadPrinterProfiles();
    void loadPrinterDirectory(const QString& path, bool system);

    QString getSystemPrinterDir() const;
    QString getUserPrinterDir() const;

    QString generateUniquePrinterId(const QString& baseId, int *outSuffix) const;

    PrinterViewData makeViewData(const PrinterProfile& profile) const; ///////////////

private:
    QMap<QString, PrinterProfile*> systemPrinters; //read only
    QMap<QString, PrinterProfile*> userPrinters; //editable & savable

    //QMap<QString, MaterialProfile> systemMaterial;
    //QMap<QString, ProcessProfile> systemProcesses;
    //QMap<QString, ProcessProfile> userProcesses;

    QString activePrinterId;
};

#endif // PROFILEMANAGER_H
