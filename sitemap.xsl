<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:sm="http://www.sitemaps.org/schemas/sitemap/0.9">
  <xsl:output method="html" indent="yes" encoding="UTF-8"/>

  <xsl:template match="/">
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>MLvisualizer Sitemap</title>
        <style>
          body { font-family: system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; padding: 20px; color: #111 }
          h1 { margin-bottom: 0.5rem }
          table { border-collapse: collapse; width: 100%; max-width: 960px }
          th, td { border: 1px solid #e3e6ee; padding: 10px; text-align: left }
          th { background: #f7f9ff }
          a { color: #0b66ff; text-decoration: none }
          a:hover { text-decoration: underline }
          .meta { color: #555; font-size: 0.95rem }
        </style>
      </head>
      <body>
        <h1>Sitemap</h1>
        <p class="meta">This page is a human-friendly view of the XML sitemap.</p>
        <table>
          <thead>
            <tr>
              <th>URL</th>
              <th>Last modified</th>
              <th>Change freq</th>
              <th>Priority</th>
            </tr>
          </thead>
          <tbody>
            <!-- use the sitemap namespace (sm) because the XML uses a default namespace -->
            <xsl:for-each select="//sm:url">
              <tr>
                <td><a href="{sm:loc}"><xsl:value-of select="sm:loc"/></a></td>
                <td><xsl:value-of select="sm:lastmod"/></td>
                <td><xsl:value-of select="sm:changefreq"/></td>
                <td><xsl:value-of select="sm:priority"/></td>
              </tr>
            </xsl:for-each>
          </tbody>
        </table>
      </body>
    </html>
  </xsl:template>

</xsl:stylesheet>
