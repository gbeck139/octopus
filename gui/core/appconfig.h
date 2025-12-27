#ifndef APPCONFIG_H
#define APPCONFIG_H

#include <QObject>
#include <QSettings>

///
/// \brief The AppConfig class stores user preferences to
/// persist settings across launches.
///
class AppConfig : public QObject
{
    Q_OBJECT
public:
    explicit AppConfig(QObject *parent = nullptr);

    bool isFirstRun() const;

    void setDefaultPrinter(int printerType);

    // User Preferences
        //default printer profile
        //default process profile
        //user preset profiles

signals:

public slots:
    void setFirstRunCompleted();
private:
    QSettings appSettings;
};

#endif // APPCONFIG_H
