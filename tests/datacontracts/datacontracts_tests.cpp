//
// Created by Francisco Nunez on 29.11.2025.
//
#include <iostream>
#include <__chrono/year_month_day.h>
#include "datacontracts/portfolio.hxx"
#include "datacontracts/Bond.hxx"


int main() {
    double coupon = 0.1;
    std::string couponFrequency = "3M";
    std::string currency = "EUR";
    auto issueDate = xml_schema::date(2026, 1, 1);
    instruments::Bond bond("bond", "FI", "ID", coupon, couponFrequency, currency);
    bond.issueDate(issueDate);

    namespace xs = ::xml_schema;
    xs::namespace_infomap ns;
    ns[""].name = "http://curveforge.com/instruments"; // default ns for <Bond>

    std::string xmlStr;
    // ---- serialize to string ----
    {
        std::ostringstream oss;
        instruments::Bond_(oss, bond, ns);
        xmlStr = oss.str();
        std::cout << xmlStr << std::endl;
    }

    // ---- deserialize from string ----
    std::istringstream iss(xmlStr);
    std::unique_ptr<instruments::Bond> ptr = instruments::Bond_(iss, xs::flags::dont_validate);

    // ---- serialize to string ----
    {
        std::ostringstream oss;
        instruments::Bond_(oss, *ptr, ns);
        xmlStr = oss.str();
        std::cout << xmlStr << std::endl;
    }
    std::cout << "XInstruments_OK";
    return 0;
}
