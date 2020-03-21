# -*- coding: utf-8 -*-
import scrapy


class BrylaSpider(scrapy.Spider):
    name = 'bryla'
    allowed_domains = ['bryla.pl']
    start_urls = ['http://bryla.pl/']

    custom_settings = {
        'DEPTH_LIMIT': 0
    }

    ARTICLES_URLS = "//section[contains(@class, 'article-list-primary row')]/article/figure/a/@href"
    NEXT_PAGE_URL = "//a[contains(@class, 'pagination__link--next')]/@href"

    TITLE_NODE = "//article[contains(@class, 'articleItem')]/h1/text()"
    LEAD_NODE = "//article[contains(@class, 'articleItem')]/div[contains(@class, 'articleItem__lead')]/p/b/text()"
    TEXT_NODE = "//section[contains(@class, 'articleSection__text typography')]//p[not(object) and not(img)]//text() | //section[contains(@class, 'articleSection__text typography')]//h2"

    def parse(self, response):
        # iterate through links to articles
        for quote in response.xpath(self.ARTICLES_URLS):
            yield response.follow(quote, callback=self.parse_article)
            
        #iterate through next page links
        for next_page in response.xpath(self.NEXT_PAGE_URL):
            yield response.follow(next_page, self.parse)
            
    def parse_article(self, response):
        yield {
            'url': response.url,
            'title': response.xpath(self.TITLE_NODE).extract(),
            'lead': response.xpath(self.LEAD_NODE).extract(),
            'text': response.xpath(self.TEXT_NODE).extract(),
        }