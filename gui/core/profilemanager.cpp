#include "profilemanager.h"

#include <QDir>
#include <QJsonDocument>
#include <QCoreApplication>

ProfileManager::ProfileManager(QObject *parent)
    : QObject{parent}
{
    // System presets (hardcoded for now)
    addPrinterProfile(PrinterProfile("cylinderOne"));
    addPrinterProfile(PrinterProfile("cylinderTwo"));
}

void ProfileManager::addPrinterProfile(const PrinterProfile &profile)
{
    printerMap.insert(profile.getId(), profile);
}

QStringList ProfileManager::getAvailablePrinters() const
{
    return printerMap.keys();
}

void ProfileManager::setActivePrinter(const QString &printerId)
{
    if (!printerMap.contains(printerId)) {
        qDebug() << "[ERROR] Unknown printer:" << printerId;
        return;
    }

    activePrinterId = printerId;
    emit activePrinterChanged(printerId);
}

QString ProfileManager::getActivePrinter()
{
    if (printerMap.contains(activePrinterId)) {
        return activePrinterId;
    } else {
        return nullptr;
    }
}

void ProfileManager::loadPrinterProfiles()
{
    printerMap.clear();

    loadPrinterDirectory(getSystemPrinterDir(), true);
    loadPrinterDirectory(getUserPrinterDir(), false);

    qDebug() << "[ProfileManager] Loaded printers:" << printerMap.keys() << ", from:" << getSystemPrinterDir() << "and" << getUserPrinterDir();
}

void ProfileManager::savePrinterProfile(const PrinterProfile &profile)
{
    //QString path = getUserPrinterDir() + "/" + profile.getId() + ".json";

    QString path = "temp";

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly))
        return;

    QJsonDocument doc(profile.toJson());
    file.write(doc.toJson(QJsonDocument::Indented));
    file.close();
}

void ProfileManager::loadPrinterDirectory(const QString &path, bool system)
{
    QDir dir(path);

    if (!dir.exists()) {
        if (!system) {
            dir.mkpath(".");
        }
        return;
    }

    QStringList files = dir.entryList({ "*.json" }, QDir::Files);

    for (const QString& fileName : files) {
        QFile file(dir.filePath(fileName));
        if (!file.open(QIODevice::ReadOnly)) {
            qDebug() << "[ERROR] Failed to open printer profile:" << fileName;
            continue;
        }

        QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
        file.close();

        if (!doc.isObject())
            continue;

        PrinterProfile profile =
            PrinterProfile::fromJson(doc.object());

        printerMap.insert(profile.getId(), profile);
    }
}

QString ProfileManager::getSystemPrinterDir() const
{
    return QCoreApplication::applicationDirPath() + "/core/profiles/system";
}

QString ProfileManager::getUserPrinterDir() const
{
    return QCoreApplication::applicationDirPath() + "/core/profiles/user";
}

