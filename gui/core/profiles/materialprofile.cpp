#include "materialprofile.h"

MaterialProfile::MaterialProfile(const QString& profileId)
    : id(profileId), displayName(profileId) {}

MaterialProfile* MaterialProfile::fromJson(const QJsonObject& obj, bool system)
{
    auto* p = new MaterialProfile(obj["id"].toString());
    p->displayName = obj["name"].toString();
    p->nozzleTemp = obj["nozzleTemp"].toInt();
    p->bedTemp = obj["bedTemp"].toInt();
    return p;
}

QJsonObject MaterialProfile::toJson() const
{
    return {
        {"id", id},
        {"name", displayName},
        {"nozzleTemp", nozzleTemp},
        {"bedTemp", bedTemp}
    };
}

MaterialProfile* MaterialProfile::clone() const
{
    auto* c = new MaterialProfile(id);
    *c = *this;
    return c;
}
