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
    void addPrinterProfile(const PrinterProfile& profile);
    QStringList getAvailablePrinters() const;
    QString getActivePrinter();

    // Save profiles
    void savePrinterProfile(const PrinterProfile& profile);

    // Active profile control (have defaults in already)

    // Lookup

public slots:
    void setActivePrinter(const QString& printerId);

signals:
    void activePrinterChanged(const QString& printerId);

private:
    // Load profiles
    void loadPrinterProfiles();
    void loadPrinterDirectory(const QString& path, bool system);

    QString getSystemPrinterDir() const;
    QString getUserPrinterDir() const;

private:
    QMap<QString, PrinterProfile> printerMap;
    //QMap<QString, MaterialProfile> materialMap;
    //QMap<QString, ProcessProfile> processMap;

    QString activePrinterId;
};

#endif // PROFILEMANAGER_H
