import React from "react";
import { Nav, NavLink } from "react-bootstrap";
import { Link, useLocation } from "react-router-dom";

const NavigationMenu = ({ links, prefix }) => {
  const location = useLocation();

  const scrollToElement = (id) => {
    const element = document.getElementById(id);
    if (element) {
      window.scrollTo({
        top: element.offsetTop - 190,
        behavior: "smooth",
      });
    }
  };

  return (
    <Nav variant="pills" className="flex-column">
      {links.map((link) => (
        <React.Fragment key={link.to}>
          <NavLink
            as={Link}
            to={`${prefix}${link.to}`}
            className={
              location.pathname === `${prefix}${link.to}`
                ? "active nav-link"
                : "nav-link"
            }
          >
            {link.label}
          </NavLink>
          {link.subLinks &&
            location.pathname === `${prefix}${link.to}` &&
            link.subLinks.map((subLink) => (
              <Nav.Link
                key={subLink.id}
                onClick={() => scrollToElement(subLink.id)}
                className="small pl-4 text-muted"
              >
                {subLink.label}
              </Nav.Link>
            ))}
        </React.Fragment>
      ))}
    </Nav>
  );
};

export default NavigationMenu;
