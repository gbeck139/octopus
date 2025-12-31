#ifndef PROFILEMANAGER_H
#define PROFILEMANAGER_H

#include <QObject>
#include <QMap>

#include "printerprofile.h"
//#include "materialprofile.h"
//#include "processprofile.h"

class ProfileManager : public QObject
{
    Q_OBJECT
public:
    explicit ProfileManager(QObject *parent = nullptr);

    // Printer profiles
    QList<const PrinterProfile*> getSystemPrinters() const;
    QList<const PrinterProfile*> getUserPrinters() const;

    QString getActivePrinter();
    PrinterProfile* getActivePrinterProfile() const;
    void addUserPrinter(PrinterProfile* profile);
    void updateUserPrinter(PrinterProfile* profile);

    // Save profiles
    void savePrinterProfile(const PrinterProfile* profile);

    // Delete User Printer << currently used for debugging purposes
    void deleteUserPrinter(const QString& id);

    // Active profile control (have defaults in already)

    // Lookup

public slots:
    void setActivePrinter(const QString& printerId);

    // void addUserPrinter(PrinterProfile* profile);
    // void updateUserPrinter(PrinterProfile* profile);

signals:
    void activePrinterChanged(const QString& printerId);
    void printersChanged(); // list structure changes
    void activePrinterDataChanged(const QString& activePrinterId); // name/limits/nozzle edited

private:
    // Load profiles
    void loadPrinterProfiles();
    void loadPrinterDirectory(const QString& path, bool system);

    QString getSystemPrinterDir() const;
    QString getUserPrinterDir() const;

    QString generateUniquePrinterId(const QString& baseId, int *outSuffix) const;

private:
    QMap<QString, PrinterProfile*> systemPrinters; //read only
    QMap<QString, PrinterProfile*> userPrinters; //editable & savable

    QString activePrinterId;

    //QMap<QString, MaterialProfile> materialMap;
    //QMap<QString, ProcessProfile> processMap;


};

#endif // PROFILEMANAGER_H
