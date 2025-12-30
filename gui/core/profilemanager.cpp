#include "profilemanager.h"

#include <QDir>
#include <QJsonDocument>
#include <QCoreApplication>

ProfileManager::ProfileManager(QObject *parent)
    : QObject{parent}
{
    loadPrinterProfiles();
}

QList<const PrinterProfile*> ProfileManager::getSystemPrinters() const
{
    QList<const PrinterProfile*> profiles;
    for (PrinterProfile* profile : systemPrinters.values()) {
        profiles.append(profile);
    }
    return profiles;
}

QList<const PrinterProfile*> ProfileManager::getUserPrinters() const
{
    QList<const PrinterProfile*> profiles;
    for (PrinterProfile* profile : userPrinters.values()) {
        profiles.append(profile);
    }
    return profiles;
}

void ProfileManager::setActivePrinter(const QString &printerId)
{
    if (!userPrinters.contains(printerId) && !systemPrinters.contains(printerId)) {
        qDebug() << "[PROFILE MANAGER] [ERROR] Unknown printer:" << printerId;
        return;
    }

    if (printerId == activePrinterId) {
        return;
    }

    activePrinterId = printerId;
    emit activePrinterChanged(printerId);
}

QString ProfileManager::getActivePrinter()
{
    return activePrinterId;
}

void ProfileManager::loadPrinterProfiles()
{
    qDeleteAll(systemPrinters);
    qDeleteAll(userPrinters);

    systemPrinters.clear();
    userPrinters.clear();

    loadPrinterDirectory(getSystemPrinterDir(), true);
    loadPrinterDirectory(getUserPrinterDir(), false);

    qDebug() << "[PROFILE MANAGER] Loaded printers:" << systemPrinters.keys() << userPrinters.keys() << ", from:" << getSystemPrinterDir() << "and" << getUserPrinterDir();
}

void ProfileManager::savePrinterProfile(const PrinterProfile &profile)
{
    //QString path = getUserPrinterDir() + "/" + profile.getId() + ".json";

    QString path = getUserPrinterDir() + "/" + profile.getId() + ".json";

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly))
        return;

    QJsonDocument doc(profile.toJson());
    file.write(doc.toJson(QJsonDocument::Indented));
    file.close();
}

void ProfileManager::loadPrinterDirectory(const QString &path, bool system)
{
    QStringList files;

    if (system) {
        // For resources, QDir can list files inside the resource
        QDir resDir(path);
        files = resDir.entryList(QStringList() << "*.json", QDir::Files);
    } else {
        // For user folder on disk
        QDir dir(path);
        if (!dir.exists()) {
            dir.mkpath(".");
        }
        files = dir.entryList(QStringList() << "*.json", QDir::Files);
    }

    for (const QString &fileName : files) {
        QString fullPath = path + "/" + fileName;
        QFile file(fullPath);
        if (!file.open(QIODevice::ReadOnly)) {
            qDebug() << "[PROFILE MANAGER] [ERROR] Failed to open printer profile:" << fullPath;
            continue;
        }

        QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
        file.close();

        if (!doc.isObject())
            continue;

        PrinterProfile* profile = PrinterProfile::fromJson(doc.object(), system);
        (system ? systemPrinters : userPrinters).insert(profile->getId(), profile);

        qDebug() << "[PROFILE MANAGER] System printers loaded:" << systemPrinters.keys();
        qDebug() << "[PROFILE MANAGER] User printers loaded:" << userPrinters.keys();
    }
}

QString ProfileManager::getSystemPrinterDir() const
{
    return ":/json/json/";
}

QString ProfileManager::getUserPrinterDir() const
{
    return QCoreApplication::applicationDirPath() + "/core/profiles/printers/user";
}

